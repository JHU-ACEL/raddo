"""Base dynamics class.

"Differentiable Model Predictive Control on the GPU" by Emre Adabag, Marcus Greiff, John Subosits, and Thomas Lew
https://github.com/ToyotaResearchInstitute/diffmpc
"""
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from typing import Any

    from jax.typing import ArrayLike


default_parameters: dict[str, Any] = {
    "num_states": 2,
    "num_controls": 2,
    "names_states": ["state_1", "state_2"],
    "names_controls": ["control_1", "control_2"],
}
default_dynamics_parameters: dict[str, Any] = {}


class Dynamics(eqx.Module):
    """Base dynamics class."""
    _params: dict[str, Any] = eqx.field(static=True)
    _dynamics_params: dict[str, Any] #= eqx.field(static=True)
    _state_name_to_state_index_dict: dict[str, Any] = eqx.field(static=True)
    _control_name_to_control_index_dict: dict[str, Any] = eqx.field(static=True)
  
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
        """
        if parameters is None:
          parameters = deepcopy(default_parameters)
        if dynamics_params is None:
          dynamics_params = deepcopy(default_dynamics_parameters)

        object.__setattr__(self, "_params", parameters)
        object.__setattr__(self, "_dynamics_params", dynamics_params)
    
        state_index_dict = {name: i for i, name in enumerate(parameters["names_states"])}
        control_index_dict = {name: i for i, name in enumerate(parameters["names_controls"])}
    
        object.__setattr__(self, "_state_name_to_state_index_dict", state_index_dict)
        object.__setattr__(self, "_control_name_to_control_index_dict", control_index_dict)

    @property
    def params(self) -> dict[str, Any]:
        """Returns the parameters of the class."""
        return self._params

    @property
    def dynamics_params(self) -> dict[str, Any]:
        """Returns the parameters of the class."""
        return self._dynamics_params

    @property
    def num_states(self) -> int:
        """Returns the number of state variables."""
        return self.params["num_states"]

    @property
    def num_controls(self) -> int:
        """Returns the number of control variables."""
        return self.params["num_controls"]

    @property
    def names_states(self) -> list[str]:
        """Returns the names of the state variables."""
        return self.params["names_states"]

    @property
    def names_controls(self) -> list[str]:
        """Returns the names of the control variables."""
        return self.params["names_controls"]

    @property
    def state_name_to_state_index_dict(self) -> dict[str, Any]:
        """Returns dictionary state_name_to_state_index_dict"""
        return self._state_name_to_state_index_dict

    @property
    def control_name_to_control_index_dict(self) -> dict[str, Any]:
        """Returns dictionary control_name_to_control_index_dict"""
        return self._control_name_to_control_index_dict

    def get_state_variable_at_state_name(
        self, state: jax.Array, state_name: str
    ) -> jax.Array:
        """
        Gets the variable named state_name in the state.

        Args:
            state: state of the system (see names_states)
                (_num_states, ) array
            state_name: name of the state variable (see names_states)
                (str)

        Returns:
            state_variable: variable named state_name in the state vector
                (float)
        """
        variable = state[self.state_name_to_state_index_dict[state_name]]
        return variable

    def get_control_variable_at_control_name(
        self, control: jax.Array, control_name: str
    ) -> jax.Array:
        """
        Gets the variable named control_name in the state.

        Args:
            control: control input of the system (see controls_states)
                (_num_controls, ) array
            control_name: name of the control variable (see controls_states)
                (str)

        Returns:
            control_variable: variable named control_name in the control vector
                (float)
        """
        variable = control[self.control_name_to_control_index_dict[control_name]]
        return variable

    def state_dot(
        self,
        state: jax.Array,
        control: jax.Array,
        t: ArrayLike = 0.0,
        external_param: jax.Array | None = None,
        disturbance_force: jax.Array | None = jnp.array([0])
        ) -> jax.Array:
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
        raise NotImplementedError

    def quaternion_projection(self, state: jax.Array) -> jax.Array:
      """ Projects quaternion onto SO(3) manifold (normalizes so norm(q) = 1. If dynamics don't have a quaternion, this done nothing"""
      return state

    def get_error_coords(self, x: jax.Array, xbar: jax.Array) -> jax.Array:
      return x - xbar
    
    def get_true_coords(self, dx: jax.Array, xbar: jax.Array) -> jax.Array:
      return dx + xbar

    # Attitude Jacobian, does nothing unless using MRP's in controller
    def E(self, x: jax.Array) -> jax.Array:
      nx = self.params["num_states"]
      return jnp.eye(nx)

    def linearize_and_discretize(
        self,
        nom_traj: jax.Array,
        nom_cntrl: jax.Array,
        ext_param: jax.Array,
        dt: ArrayLike,
    ) -> tuple[jax.Array, jax.Array]:
      nx = self.params["num_states"]
      def linearize_and_discretize_single(
          x: jax.Array,
          xp: jax.Array,
          u: jax.Array,
          ext_param: jax.Array,
      ) -> tuple[jax.Array, jax.Array]:
        A = jax.jacfwd(self.state_dot,argnums=0)(x,u,0.0,ext_param) #df/dx
        B = jax.jacfwd(self.state_dot,argnums=1)(x,u,0.0,ext_param) # df/du
  
        # xp is x_k+1, transformation in "Planning with Attitude"
        A = self.E(xp).T @ A @ self.E(x)
        B = self.E(xp).T @ B
        
        Ad = jnp.eye(nx) + A*dt 
        Bd = B*dt
  
        return Ad, Bd
        
      Ad, Bd = jax.vmap(linearize_and_discretize_single)(nom_traj[:-1], nom_traj[1:], nom_cntrl, ext_param)
  
      return Ad, Bd

      