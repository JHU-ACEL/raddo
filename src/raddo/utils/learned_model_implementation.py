from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from .coord_transforms import coord

if TYPE_CHECKING:
    from typing import Any

    from jax.typing import ArrayLike

    from raddo.dynamics.planetary_params import PlanetParams


""" 
This file sets up classes for applying learned models within the TrajectoryGenerator class. 
By defining a new class and passing to the TrajectoryGenerator, you can specify how learning is used within the trajectory rollout 
(e.g. using NN magnetic field model to produce b-field values from orbital position
or perform online adaptation of magnetic field model using new magnetometer measurements)

Setting up this way allows easily adding future implementation of learning by defining in same
structure as the base LearningApplicator class
"""

class LearningApplicator(eqx.Module):
    """Base class applying the learned model within the TrajectoryGenerator class
    
    The key contract:
        - `init_state()` returns any carry state needed across timesteps
        - `__call__()` takes the current info and returns estimated params
          for the controller horizon, plus updated carry state
    """
    
    @abc.abstractmethod
    def init_state(self) -> Any:
        """Return initial carry state (e.g., initial model weights)."""
        ...
    
    @abc.abstractmethod
    def __call__(
        self,
        carry: Any,
        dyn_params_est_slice: jax.Array,   
        dyn_params_est_slice_history: jax.Array,   
        i: int,
        dt: ArrayLike,
        horizon: int,
        key: jax.Array
    ) -> tuple[jax.Array, Any]:
        """
        Returns:
            estimated_params: (horizon, params_dim) — params to pass to controller
            new_carry: updated carry state
        """
        ...

class LearnedPosToMagMap(LearningApplicator):
    model: eqx.Module
    planet: Any

    def init_state(self) -> eqx.Module:
        return self.model
      
    def __call__(
        self,
        carry: eqx.Module,
        dyn_params_est_slice: jax.Array,
        dyn_params_est_slice_history: jax.Array,
        i: int | jax.Array,
        dt: ArrayLike,
        horizon: int,
        key: jax.Array,
    ) -> tuple[jax.Array, eqx.Module]:
        """Compute B-field from learned model with orbital position as inputs to pass to controller """
        model = carry
        position_pci = dyn_params_est_slice[:,:3]
        b_field_pci = _pci_pos_to_b(i, dt, horizon, position_pci, model, self.planet)
        
        return b_field_pci, model

class AdaptMagModel(LearningApplicator):
    """Learned model that adapts online using measurement history."""
    model: eqx.Module
    planet: Any
    adapt_fn: callable
    adapt_freq: int
    history: int
    adapt_batch_size: int
    adapt_lr: float
    adapt_iter: int

    def init_state(self) -> eqx.Module:
        return self.model

    def __call__(
        self,
        carry: eqx.Module,
        dyn_params_est_slice: jax.Array,
        dyn_params_est_slice_history: jax.Array,
        i: int | jax.Array,
        dt: ArrayLike,
        horizon: int,
        key: jax.Array,
    ) -> tuple[jax.Array, eqx.Module]:
        model = carry
        # Adaptation
        # Get history of position and mag measurments
        position_pci_hist = dyn_params_est_slice_history[:,:3]
        magnetometer_measurements = dyn_params_est_slice_history[:,3:]
        
        # Convert positions to 4D spherical coords
        t_hist_slice = (i-self.history)*dt + dt*jnp.arange(self.history)
        pcpf_positions = coord.pci_to_pcpf(position_pci_hist, t_hist_slice, self.planet)
        spherical_positions_history = coord.cartesian_to_spherical_4D(pcpf_positions)
        
        def do_adapt(operand: tuple[Any, ...]) -> eqx.Module:
              return self.adapt_fn(
                  operand[0], operand[1], operand[2],
                  self.adapt_batch_size, self.adapt_lr, self.adapt_iter, key)
          
        def no_adapt(operand: tuple[Any, ...]) -> eqx.Module:
              return operand[0]
  
        # Only adapt every adapt_freq steps, and only after history steps
        model = jax.lax.cond(
            (i % self.adapt_freq == 0) & (i >= self.history),
            do_adapt,
            no_adapt,
            operand=(model, spherical_positions_history, magnetometer_measurements)
        )

        # Use adapted learned model
        position_pci = dyn_params_est_slice[:,:3]
        b_field_pci = _pci_pos_to_b(i, dt, horizon, position_pci, model, self.planet)
  
        return b_field_pci, model
    

def _pci_pos_to_b(
    i: int | jax.Array,
    dt: ArrayLike,
    horizon: int,
    position_pci: jax.Array,
    model: eqx.Module,
    planet: PlanetParams,
) -> jax.Array:
    # Convert XYZ in PCI to PCPF
    t_slice = i*dt + dt*jnp.arange(horizon)
    dyn_params_est_slice = coord.pci_to_pcpf(position_pci, t_slice, planet) 
    # Convert dyn_params_est_slice (xyz pos) to PCPF -> 4D spherical
    dyn_params_est_slice = coord.cartesian_to_spherical_4D(dyn_params_est_slice)
    # Compute b-field (4D spherical -> B-field in PCPF)
    dyn_params_est_slice = jax.vmap(model)(dyn_params_est_slice)
    # Convert b-field back to PCI and save as dyn_params_est_slice
    dyn_params_est_slice = coord.pcpf_to_pci(dyn_params_est_slice, t_slice, planet)
    return dyn_params_est_slice
