from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from .base_controller import Controller

if TYPE_CHECKING:
    from raddo.dynamics.base_dynamics import Dynamics
    from raddo.utils.propagate import TrajectoryGenerator


class MPPIController(Controller):
  propagator: TrajectoryGenerator
  dynamics: Dynamics
  
  # Parameters
  Q: jax.Array
  Qf: jax.Array
  R: jax.Array
  control_limits: jax.Array
  
  # Hyperparameters (Static) (ints and floats default to static)
  num_samples: int
  num_iter: int
  horizon: int
  kappa: float

  # Useful for working with dynamics
  quat_start: int
  nx: int
  nu: int

  def __init__(
      self,
      propagator: TrajectoryGenerator,
      Q: jax.Array,
      Qf: jax.Array,
      R: jax.Array,
      control_limits: jax.Array,
      num_samples: int,
      num_iter: int,
      horizon: int,
      kappa: float,
  ) -> None:
    object.__setattr__(self, "propagator", propagator)
    object.__setattr__(self, "dynamics", self.propagator.dynamics)
    object.__setattr__(self, "Q", (Q + Q.T)/2) # ensure symmetric
    object.__setattr__(self, "Qf", (Qf + Qf.T)/2)
    object.__setattr__(self, "R", (R+R.T)/2)
    object.__setattr__(self, "num_samples", num_samples)
    object.__setattr__(self, "num_iter", num_iter)
    object.__setattr__(self, "horizon", horizon)
    object.__setattr__(self, "kappa", kappa)
    object.__setattr__(self, "nu", self.dynamics.num_controls)
    object.__setattr__(self, "control_limits", control_limits)

    self.nx = self.dynamics.num_states
    nx = self.dynamics.num_states
    quat_start = self.dynamics.params.get("quat_start")

    # Check if attitude Jacobian is square (whether using mrp's or not)
    if not self.dynamics.E(jnp.ones(nx)).shape[0] == self.dynamics.E(jnp.ones(nx)).shape[1]:
      nx = self.dynamics.num_states - 1

    object.__setattr__(self, "nx", nx)

    object.__setattr__(self, "quat_start", quat_start)
    
  def compute_cost(
      self,
      states: jax.Array,
      controls: jax.Array,
      target_state: jax.Array,
  ) -> jax.Array:
    """Computes the total quadratic cost for a trajectory and control sequence, including terminal cost.
        Inputs:
        states: Sequence of states, shape (T+1, nx)
        controls: Sequence of controls, shape (T, nu)
        Outputs:
        cost: Scalar total cost (float)
    """
    # Terminal cost
    final_error = self.dynamics.get_error_coordinates(states[-1,:], target_state)
    J_terminal = 0.5 * final_error.T @ self.Qf @ final_error

    # Stage cost (vectorized over time) 
    diff_stage = states[:-1, :] - target_state
    
    def stage_step_cost(x_k: jax.Array, u_k: jax.Array) -> jax.Array:
      error = self.dynamics.get_error_coordinates(x_k, target_state) 
      return (0.5 * error.T @ self.Q @ error + 0.5 * u_k.T @ self.R @ u_k)*self.propagator.dt
        
    J_stage = jax.vmap(stage_step_cost)(states[:-1], controls)
    
    return J_terminal + jnp.sum(J_stage)
    
  @eqx.filter_jit
  def __call__(
      self,
      x0: jax.Array,
      x_goal: jax.Array,
      key: jax.Array,
      external_dynamics_params: jax.Array,
      nominal_traj: jax.Array | None=None,
      nominal_cntrl: jax.Array | None=None,
  ) -> tuple[jax.Array, jax.Array]:
    """
    MPPI algorithm.
    Returns:
        nominal state trajectory: (T+1, nx)
        nominal control trajectory: (T, nu)
    """
    # Get appropriate standard deviations for noise to be added
    control_range = self.control_limits[:, 1] - self.control_limits[:, 0]
    noise_std = jnp.abs(control_range) / 10.0

    # Initialize nominal control sequence to zeros
    u_nominal = jnp.zeros((self.horizon, self.dynamics.num_controls))
    
    # Optimization Loop
    def scan_step(
        carry: tuple[jax.Array, jax.Array],
        _: None,
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        u_nom, key = carry
        
        # Split key for noise generation
        key, noise_key, traj_key = jrandom.split(key, 3)
        
        # Generate noise: (num_samples, T, nu)
        noise = jrandom.normal(noise_key, shape=(self.num_samples, self.horizon, self.dynamics.num_controls)) * noise_std
        
        # Create perturbed control sequences
        u_candidates = u_nom + noise # u_nom broadcasted across num_samples dimension
        
        # Clip controls
        u_candidates = jnp.clip(u_candidates, self.control_limits[:, 0], self.control_limits[:, 1])

        # Rollout batch of trajectories from different u_candidates
        states_batch, _ = self.propagator._generate_trajectory_batch_seq(
                                initial_states = x0,
                                target_states = x_goal,
                                key = traj_key,
                                batch_size = self.num_samples,
                                noise_std = 0.0,
                                num_steps = self.horizon,
                                external_dynamics_params = external_dynamics_params,
                                control_sequence = u_candidates)
        
        # Compute costs (vmap over the batch dimension)
        costs = jax.vmap(self.compute_cost, in_axes=(0, 0, None))(
            states_batch, u_candidates, x_goal)

        # Compute weights
        min_cost = jnp.min(costs)
        exp_costs = jnp.exp(-1.0 / self.kappa * (costs - min_cost))
        denom = jnp.sum(exp_costs) + 1e-10 
        weights = exp_costs / denom
        
        # Update nominal control: weighted sum of candidates
        # weights: (K,), u_candidates: (K, T, nu) -> sum over K -> (T, nu)
        u_update = jnp.sum(weights[:, None, None] * u_candidates, axis=0)
        
        u_new = jnp.clip(u_update, self.control_limits[:, 0], self.control_limits[:, 1])
        
        return (u_new, key), carry

    # Run optimization
    init_carry = (u_nominal, key)
    (final_u_nominal, final_key), _ = jax.lax.scan(scan_step, init_carry, length=self.num_iter)

    best_states, best_controls = self.propagator._generate_trajectory_seq(
                                initial_state = x0,
                                target_state = x_goal,
                                key = final_key,
                                noise_std = 0.0,
                                num_steps = self.horizon,
                                external_dynamics_params = external_dynamics_params,
                                control_sequence = final_u_nominal)
    

    return best_states, best_controls
