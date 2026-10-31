from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

from .base_controller import Controller

if TYPE_CHECKING:
    import jax

    from raddo.dynamics.base_dynamics import Dynamics
    from raddo.utils.propagate import TrajectoryGenerator


class PIDController(Controller):
  dynamics: Dynamics
  cntrl_param_init: jax.Array
  
  # Parameters
  control_limits: jax.Array
  kp: int
  kd: int
  ki: int

  def __init__(
      self,
      propagator: TrajectoryGenerator,
      kp: float,
      kd: float,
      ki: float,
      control_limits: jax.Array,
  ) -> None:
    self.dynamics = propagator.dynamics
    self.cntrl_param_init = jnp.zeros(3) # initial error
    self.control_limits = control_limits
    self.kp = kp
    self.kd = kd
    self.ki = ki


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
    sum_prev_error = cntrl_param
    dx0 = self.dynamics.get_error_coords(x0, nominal_traj[0])
    tau_desired = -self.kp*dx0[:3] - self.kd*dx0[3:] - self.ki*sum_prev_error
    u = jnp.cross(external_dynamics_params[0], tau_desired)

    sum_error = dx0[:3] + sum_prev_error
    

    # Clip controls
    u = jnp.clip(u, self.control_limits[:, 0], self.control_limits[:, 1])

    return u, sum_error
