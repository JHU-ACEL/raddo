"""Quadrotor dynamics class. With rotation defined using a right-handed quaternion (not JPL convention)

"Differentiable Model Predictive Control on the GPU" by Emre Adabag, Marcus Greiff, John Subosits, and Thomas Lew
https://github.com/ToyotaResearchInstitute/diffmpc
"""
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import jax.numpy as jnp

from .base_dynamics import Dynamics

if TYPE_CHECKING:
    from typing import Any

    import jax
    from jax.typing import ArrayLike


double_integrator_parameters: dict[str, Any] = {
    "num_states": 4,
    "num_controls": 2,
    "names_states": [
        "pos_x",
        "pos_y",
        "vel_x",
        "vel_y",
    ],
    "names_controls": ["force_x", "force_y"],
}
double_integrator_dynamics_parameters = {
    "mass": 0.1,
}

class DoubleIntegratorDynamics(Dynamics):
    """2D double integrator dynamics class."""

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
            parameters = deepcopy(double_integrator_parameters)
        if dynamics_params is None:
            dynamics_params = deepcopy(double_integrator_dynamics_parameters)

        super().__init__(parameters, dynamics_params)

    def state_dot(
        self,
        state: jax.Array,
        control: jax.Array,
        t: ArrayLike = 0.0,
        external_param: jax.Array | None = None,
        disturbance_force: jax.Array | None = jnp.zeros(2)) -> jax.Array:
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
        m = self.dynamics_params["mass"] # kg
        x, y, x_dot, y_dot = state
        u_x, u_y = control + disturbance_force
    
        # Dynamics
        x_ddot = u_x/m 
        y_ddot = u_y/m
    
        state_dot = jnp.array([x_dot, y_dot, x_ddot, y_ddot])
        return state_dot
