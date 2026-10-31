from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

from .base_controller import Controller

if TYPE_CHECKING:
    import jax

    from raddo.dynamics.base_dynamics import Dynamics
    from raddo.utils.propagate import TrajectoryGenerator


class FeedbackController(Controller):
  propagator: TrajectoryGenerator
  dynamics: Dynamics
  cntrl_param_init: jax.Array
  
  # Parameters
  Q: jax.Array
  R: jax.Array
  control_limits: jax.Array

  nx: int = eqx.field(static=True)

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

    # Check if attitude Jacobian is square (whether using mrp's or not)
    if not self.dynamics.E(jnp.ones(nx)).shape[0] == self.dynamics.E(jnp.ones(nx)).shape[1]:
      nx = self.dynamics.num_states - 1

    object.__setattr__(self, "nx", nx)


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
        nominal state trajectory: (T+1, nx)
        nominal control trajectory: (T, nu)
    """
    dt = self.propagator.dt
    A, B = self.dynamics.linearize_and_discretize(jnp.array([x0, x0]), nominal_cntrl[:1], external_dynamics_params[:1], dt)
    #A, B = self.dynamics.linearize_and_discretize(nominal_traj, nominal_cntrl, external_dynamics_params, dt)
    
    A = A[0]
    B = B[0]
    Q = self.Q
    R = self.R 
    S = cntrl_param

    dx = self.dynamics.get_error_coords(x0, nominal_traj[0])

    # Ricatti equation
    eps = 1e-8*jnp.ones(R.shape) # to avoid singular matrix
    K = jnp.linalg.solve(R + B.T @ S @ B + eps, B.T @ S @ A)
    # S = Q + K.T @ R @ K + (A - B@K).T @ S @ (A - B@K)
    S = Q + A.T @ S @ A - A.T @ S @ B @ K

    # Control law
    u = nominal_cntrl[0] - K @ dx
    
    # Clip controls
    u = jnp.clip(u, self.control_limits[:, 0], self.control_limits[:, 1])

    return u, S
