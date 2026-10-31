"""Spacecraft dynamics class. With rotation defined using a right-handed quaternion (not JPL convention)

Setup from "Differentiable Model Predictive Control on the GPU" by Emre Adabag, Marcus Greiff, John Subosits, and Thomas Lew
https://github.com/ToyotaResearchInstitute/diffmpc
"""
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from .base_dynamics import Dynamics
from .quaternion_functions import S, get_rotation, q_conj, q_left, QuatMRP #,mrp_to_q, q_to_mrp

if TYPE_CHECKING:
    from typing import Any
    from jax.typing import ArrayLike


spacecraft_parameters: dict[str, Any] = {
    "num_states": 7,
    "num_controls": 3,
    "names_states": [
        "q_0", # scalar part
        "q_1",
        "q_2",
        "q_3",
        "omega_x",
        "omega_y",
        "omega_z",
    ],
    "names_controls": ["dipole_x", "dipole_y", "dipole_z"], # [A m^2]
    "quat_start": 0,
}

# 1U properties
spacecraft_dynamics_parameters = {
    "mass": 0.75,
    "inertia": jnp.array([0.00125, 0.0001, 0.0001, 0.0001, 0.00125, 0.0001, 0.0001, 0.0001, 0.00125]).reshape((3, 3)),
}
spacecraft_dynamics_parameters["inertia_inv"] = jnp.linalg.inv(spacecraft_dynamics_parameters["inertia"]) 

class MagnetorquerDynamics(QuatMRP, Dynamics):
    """Magnetorquer control spacecraft dynamics class."""
    def __init__(
        self,
        parameters: dict[str, Any] | None=None,
        dynamics_params: dict[str, Any] | None=None,
    ) -> None:
      """
      Initializes the class.
      Args:
          parameters:  parameters of the class.
              (str, Any) dictionary
          dynamics_parameters:  parameters for the dynamics of the class.
              (str, Any) dictionary
      """

      if parameters is None:
          parameters = deepcopy(spacecraft_parameters)
      if dynamics_params is None:
          dynamics_params = deepcopy(spacecraft_dynamics_parameters)
    
      super().__init__(parameters, dynamics_params)
      
    def state_dot(
        self,
        state: jax.Array,
        control: jax.Array,
        t: ArrayLike = 0.0,
        external_param: jax.Array | None = None,
        disturbance_force: jax.Array | None = jnp.array(3)) -> jax.Array:
        """
        Computes the time derivative of the state of the system.

        Returns x_dot = f(x, u) where f describes the dynamics of the system.

        Args:
            state: state of the system (see names_states)
                (_num_states, ) array
            control: control input applied to the system (see names_controls)
                (_num_controls, ) array
            params: parameters of the state_dot function of the dynamics.
                (str, Any) dictionary

        Returns:
            state_dot: time derivative of the state
                (_num_states, ) array
        """
        inertia = self.dynamics_params["inertia"]
        inertia_inverse = self.dynamics_params["inertia_inv"]

        # Get b-field (planet-centered inertial coordinate frame)
        # Learned model is applied in LearningApplicator, this makes it so these dynamics
        # with and without learning (for true dynamics propagation)
        b_pci = external_param 
      
        # Extract states
        q = jnp.array(state[0:4]) 
        w = jnp.array(state[4:7]) # [rad/s]
        dipole = control[0:3] # [A m^2] magnetic dipoles are in body frame
      
        # Magnetorquer torque
        b_body = get_rotation(q_conj(q)) @ b_pci # using q_conj goes from pci to body
        tau = jnp.cross(dipole, b_body)*1e-9 + disturbance_force # convert nT to T, A*m^2 * T results in Nm

        # Calculate angular velocities and accelerations
        # Conventions/formulation from "Planning with Attitude" - Jackson et. al.
        q_dot = 0.5 * q_left(q) @ jnp.concatenate((jnp.array([0.0]), w))
        w_dot = inertia_inverse @ (tau - S(w)@ inertia@w) # cross product a x b = a_skew_symmetric @ b
      
        state_dot = jnp.concatenate((q_dot, w_dot))
        return state_dot

    def quaternion_projection(self, state: jax.Array) -> jax.Array:
      quat_start = self.params["quat_start"]
      q = state[quat_start:quat_start+4]
      return state.at[quat_start:quat_start+4].set(q / jnp.linalg.norm(q))

    # def get_error_coords(self, x: jax.Array, xbar: jax.Array) -> jax.Array:
    #   qs = self.params["quat_start"]
    #   # Linear error subtraction for states before and after the 4-element quaternion
    #   d_pre = x[:qs] - xbar[:qs]
    #   dq = q_to_mrp(x[qs : qs + 4], xbar[qs : qs + 4])
    #   d_post = x[qs + 4 :] - xbar[qs + 4 :]
    #   return jnp.concatenate((d_pre, dq, d_post))
    
    # def get_true_coords(self, dx: jax.Array, xbar: jax.Array) -> jax.Array:
    #   qs = self.params["quat_start"]
    #   # Linear addition for states before and after the 3-element MRP error
    #   x_pre = dx[:qs] + xbar[:qs]
    #   q = mrp_to_q(dx[qs : qs + 3], xbar[qs : qs + 4])
    #   x_post = dx[qs + 3 :] + xbar[qs + 4 :]
    #   return jnp.concatenate((x_pre, q, x_post))
    
    # def E(self, x: jax.Array) -> jax.Array:
    #   nx = x.shape[0]
    #   qs = self.params["quat_start"]
  
    #   # Extract quaternion components starting at quat_start
    #   q1, q2, q3, q4 = x[qs], x[qs + 1], x[qs + 2], x[qs + 3]
    #   G = jnp.array([[-q2, -q3, -q4],
    #                 [q1, -q4, q3],
    #                 [q4, q1, -q2],
    #                 [-q3, q2, q1],])
  
    #   E_mat = jnp.zeros((nx, nx - 1))
    #   # Identity block for states before the quaternion
    #   E_mat = E_mat.at[:qs, :qs].set(jnp.eye(qs))
    #   # Kinematic mapping block for the 4-element quaternion to 3-element MRP
    #   E_mat = E_mat.at[qs : qs + 4, qs : qs + 3].set(G)
    #   # Identity block for all remaining states after the quaternion
    #   E_mat = E_mat.at[qs + 4 :, qs + 3 :].set(jnp.eye(nx - qs - 4))
    #   return E_mat

    # def linearize_and_discretize(
    #     self,
    #     nom_traj: jax.Array,
    #     nom_cntrl: jax.Array,
    #     ext_param: jax.Array,
    #     dt: ArrayLike,
    # ) -> tuple[jax.Array, jax.Array]:
    #   nx = self.params["num_states"]-1
    #   df_dx = jax.jacfwd(self.state_dot, argnums=0)
    #   df_du = jax.jacfwd(self.state_dot, argnums=1)
      
    #   def linearize_and_discretize_single(
    #       x: jax.Array,
    #       xp: jax.Array,
    #       u: jax.Array,
    #       ext_param: jax.Array,
    #   ) -> tuple[jax.Array, jax.Array]:
    #     A = df_dx(x,u,0.0,ext_param) #df/dx
    #     B = df_du(x,u,0.0,ext_param) # df/du
  
    #     # xp is x_k+1, transformation in "Planning with Attitude"
    #     A = self.E(xp).T @ A @ self.E(x)
    #     B = self.E(xp).T @ B
        
    #     Ad = jnp.eye(nx) + A*dt 
    #     Bd = B*dt
  
    #     return Ad, Bd
        
    #   Ad, Bd = jax.vmap(linearize_and_discretize_single)(nom_traj[:-1], nom_traj[1:], nom_cntrl, ext_param)
  
    #   return Ad, Bd