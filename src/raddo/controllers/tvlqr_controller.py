from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from .base_controller import Controller

if TYPE_CHECKING:
    from raddo.dynamics.base_dynamics import Dynamics
    from raddo.utils.propagate import TrajectoryGenerator


class TVLQRController(Controller):
  propagator: TrajectoryGenerator
  dynamics: Dynamics
  cntrl_param_init: jax.Array
  
  # Parameters
  Q: jax.Array
  R: jax.Array
  control_limits: jax.Array

  nx: int = eqx.field(static=True)
  nu: int = eqx.field(static=True)

  def __init__(
      self,
      propagator: TrajectoryGenerator,
      Q: jax.Array,
      Qf: jax.Array,
      R: jax.Array,
      control_limits: jax.Array,
  ) -> None:
    self.propagator = propagator
    self.dynamics = self.propagator.dynamics
    self.Q = Q
    self.cntrl_param_init = Qf
    self.R = R
    self.control_limits = control_limits

    nx = self.dynamics.num_states
    nu = self.dynamics.num_controls

    # Check if attitude Jacobian is square (whether using mrp's or not)
    if not self.dynamics.E(jnp.ones(nx)).shape[0] == self.dynamics.E(jnp.ones(nx)).shape[1]:
      nx = self.dynamics.num_states - 1

    object.__setattr__(self, "nx", nx)
    object.__setattr__(self, "nu", nu)


  @eqx.filter_jit
  def __call__(
      self,
      x0: jax.Array,
      x_goal: jax.Array,
      key: jax.Array,
      external_dynamics_params: jax.Array,
      nominal_traj: jax.Array,
      nominal_cntrl: jax.Array,
      cntrl_param: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    """
    Feedback controller for tracking a nominal trajectory

    Returns:
        nominal state trajectory: (N+1, nx)
        nominal control trajectory: (N, nu)
    """
    dt = self.propagator.dt
    A, B = self.dynamics.linearize_and_discretize(nominal_traj, nominal_cntrl, external_dynamics_params, dt)
    Q = self.Q
    eps = 1e-8*jnp.eye(self.nu) # to avoid singular matrix
    R = self.R + eps
    S_init = self.cntrl_param_init # (Qf)

    dx0 = self.dynamics.get_error_coords(x0, nominal_traj[0])

    # Flip so we are starting from end of traj
    scan_inputs = (A[::-1], B[::-1])

    def backward_pass(
        S_next: jax.Array,
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
      A_k, B_k = inputs

      # Ricatti equation 
      K_k = jnp.linalg.solve(R + B_k.T @ S_next @ B_k, B_k.T @ S_next @ A_k)
      #S_k = Q + K_k.T @ R @ K_k + (A_k - B_k@K_k).T @ S_next @ (A_k - B_k@K_k)
      S_k = Q + A_k.T @ S_next @ A_k - A_k.T @ S_next @ B_k @ K_k

      return S_k, (S_k, K_k)

    _, (S_backwards, K_backwards) = jax.lax.scan(backward_pass, init=S_init, xs=scan_inputs)

    # Get control input for current timestep
    K0 = K_backwards[-1]
    u = nominal_cntrl[0] - K0 @ dx0
    
    # Clip controls
    u = jnp.clip(u, self.control_limits[:, 0], self.control_limits[:, 1])

    # return nominal_cntrl[0], cntrl_param
    return u, S_backwards[-1]
