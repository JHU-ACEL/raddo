"""Spacecraft dynamics class. With rotation defined using a right-handed quaternion (not JPL convention)

Setup from "Differentiable Model Predictive Control on the GPU" by Emre Adabag, Marcus Greiff, John Subosits, and Thomas Lew
https://github.com/ToyotaResearchInstitute/diffmpc
"""
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax.random as jrandom

from .base_dynamics import Dynamics
from .magnetic_field import MagneticFieldModel
from .planetary_params import Earth

if TYPE_CHECKING:
    from typing import Any

    import jax
    from jax.typing import ArrayLike

    from .planetary_params import PlanetParams


orbit_parameters: dict[str, Any] = {
    "num_states": 6,
    "num_controls": 3,
    "names_states": [
        "pos_x",
        "pos_y",
        "pos_z",
        "vel_x",
        "vel_y",
        "vel_z",
    ],
    "names_controls": ["thrust_x", "thrust_y", "thrust_z"], #[kN]
}
orbit_dynamics_parameters = {
    "mass": 0.1,
    "inertia": jnp.array([0.1, 0.01, 0.01, 0.01, 0.1, 0.01, 0.01, 0.01, 0.1]),
}


class OrbitDynamics(Dynamics):
    """Orbit dynamics class for generating orbit/magnetic field data"""
    mag_model: MagneticFieldModel
    planet: PlanetParams

    def __init__(
        self,
        parameters: dict[str, Any] | None=None,
        dynamics_params: dict[str, Any] | None=None,
        Planet: PlanetParams | None=None,
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
            parameters = deepcopy(orbit_parameters)
        if dynamics_params is None:
            dynamics_params = deepcopy(orbit_dynamics_parameters)

        # Default to Earth
        if Planet is None:
          Planet = Earth
        object.__setattr__(self, "mag_model", MagneticFieldModel(Planet))
        object.__setattr__(self, "planet", Planet)

        super().__init__(parameters, dynamics_params)
      
    def state_dot(
        self,
        state: jax.Array,
        control: jax.Array,
        t: ArrayLike = 0.0,
        external_param: jax.Array | None = None,
        disturbance_force: jax.Array | None = jnp.array([0])) -> jax.Array:
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
        # Earth's gravitational parameter
        mu = self.planet.mu
        J2 = self.planet.J2
      
        inertia = self.dynamics_params["inertia"].reshape((3, 3))
        inertia_inverse = jnp.linalg.inv(inertia)  # We might want to do this computation elsewhere.
        m = self.dynamics_params["mass"]
      
        # Extract states
        r = jnp.array(state[0:3]) # [km] (ECI frame)
        v = jnp.array(state[3:6]) # [km/s] (ECI frame)
        f = control[0:3] # [kN] thrusts are in global frame (ECI frame)
      
        # Calculate gravitational acceleration
        r_mag = jnp.linalg.norm(r)
        ax_grav = -mu * r[0] / r_mag**3 + (J2*r[0] / r_mag**7) * (6*r[2]**2 - 1.5*(r[0]**2 + r[1]**2))
        ay_grav = -mu * r[1] / r_mag**3 + (J2*r[1] / r_mag**7) * (6*r[2]**2 - 1.5*(r[0]**2 + r[1]**2))
        az_grav = -mu * r[2] / r_mag**3 + (J2*r[2] / r_mag**7) * (3*r[2]**2 - 4.5*(r[0]**2 + r[1]**2))
      
        # Calculate accelerations
        v_dot = jnp.array([ax_grav, ay_grav, az_grav]) + f/m
        p_dot = v
      
        state_dot = jnp.concatenate((p_dot, v_dot))
        return state_dot

    def get_magnetometer_reading(
        self,
        state: jax.Array,
        t: ArrayLike,
        key: jax.Array,
        noise_std: ArrayLike,
        bias: ArrayLike,
        mag_model: MagneticFieldModel | None=None,
    ) -> jax.Array:
        """
        Simulates a magnetometer reading in the PCI frame [nT].
        
        Args:
            state: The current spacecraft state (contains quaternion q at state[6:10])
            t: time since epoch (seconds)
            key: JAX random key for noise generation
            noise_std: Standard deviation of sensor noise (nT)
            bias: jnp.array(3,) constant offset (nT)
        """
        if mag_model is None:
          mag_model = self.mag_model
        state = jnp.atleast_2d(state)
        r = state[:,:3]

        # True Physics
        b_pci = mag_model.compute_b_pci(r, t) # [nT]
        #b_body = get_rotation(q_conj(q)) @ b_eci

        if bias is not None:
          b_pci += bias
        
        # Sensor Noise
        noise = jrandom.normal(key, shape=(3,)) * noise_std
        return b_pci + noise
