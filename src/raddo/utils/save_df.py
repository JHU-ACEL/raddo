from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import jax
    from jax.typing import ArrayLike
    from pandas import DataFrame


def save_data_to_pd_df(
    trajectories: jax.Array,
    target_states: jax.Array,
    filename: str | None=None,
    labels: Sequence[str] | None=None,
    dt: ArrayLike=0.1,
    quat_start: int=0,
) -> DataFrame:
  """ Save results from multiple different Monte Carlo sims to one Pandas dataframe """
  try:
      import pandas as pd
  except ImportError as exc:
      raise ImportError(
          "save_data_to_pd_df requires optional dependencies: pip install 'raddo[extras]'"
      ) from exc


  # Normalize dimensions to 4D: [groups, n_trajs, time_steps, state_dim]
  if trajectories.ndim == 2:
      trajectories = trajectories[None, None, :, :]
  elif trajectories.ndim == 3:
      trajectories = trajectories[None, :, :, :]
  
  n_groups = trajectories.shape[0]
  batch_size = trajectories.shape[1]
  n_steps = trajectories.shape[2]

  # horizon_N = trajectory.shape[1]
  time = jnp.arange(n_steps) * dt
  time_expanded = jnp.broadcast_to(time, (batch_size,n_steps))

  # Normalize target_states to 3D: [groups, n_trajs, state_dim]
  if target_states.ndim == 1:
      target_states = target_states[None, None, :]
  elif target_states.ndim == 2:
      target_states = target_states[None, :, :]

  # Process data and compile into Pandas DataFrame for Seaborn
  all_data = []
  
  for g in range(n_groups):
      trajs = trajectories[g]
      # Match targets to this group
      t_states = target_states[g] if target_states.shape[0] == n_groups else target_states[0]

      # Calculate Costs
      quats = trajs[:, :, quat_start:quat_start+4]
      quat_goals = t_states[:, None, :4]
      quat_costs = 1.0 - jnp.abs(jnp.sum(quat_goals * quats, axis=-1))

      principal_rotation_angles = 2*jnp.arccos(jnp.abs(jnp.sum(quat_goals * quats, axis=-1)))*180/jnp.pi
      
      omegas = trajs[:, :, 4:]
      omega_goals = t_states[:, None, 4:]
      omega_norms = jnp.linalg.norm(omegas - omega_goals, axis=-1)*180/jnp.pi

      if labels is None:
        group_label = f"Group_{g+1}" if n_groups > 1 else "Trajectories"
      else:
        group_label = labels[g]
      
      # Append to DataFrame structure
      df_group = pd.DataFrame({
          "Group": group_label,
          "Time": np.array(time_expanded.ravel()),
          "Quaternion Errors": np.array(quat_costs.ravel()),
          "Angle Errors": np.array(principal_rotation_angles.ravel()),
          "Omega Errors": np.array(omega_norms.ravel()),
      })
      all_data.append(df_group)
      
  # Merge all groups into one DataFrame
  df = pd.concat(all_data, ignore_index=True)
  if filename is not None:
    from pathlib import Path

    # Create data folder
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Now safe to save
    filepath = data_dir / filename
    df.to_pickle(filepath)
  
  return df