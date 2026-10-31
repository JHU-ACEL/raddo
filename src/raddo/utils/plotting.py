from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pathlib import Path

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray
    from pandas import DataFrame


def create_stats_df(
    df: DataFrame,
    batch_size: int,
    N: int,
    angle_tol: float,
    omega_tol: float,
    tail_length: int,
    labels: Sequence[str] | None=None,
) -> DataFrame:
  try:
      import pandas as pd
  except ImportError as exc:
      raise ImportError(
          "create_stats_df requires optional dependencies: pip install 'raddo[extras]'"
      ) from exc

  n_groups = df["Group"].nunique()

  # 2. Process data and compile into Pandas DataFrame for Seaborn
  all_data = []
  
  for g in range(n_groups):
    group = df["Group"].unique()[g]
    group_df = df.loc[df["Group"] == group]
    time_vector = group_df["Time"].to_numpy(dtype=float).reshape((batch_size, N))
    time_vector = time_vector[0]
    dt = time_vector[1] - time_vector[0]
    quat_costs = group_df["Quaternion Errors"].to_numpy(dtype=float).reshape((batch_size, N))
    principal_rotation_angles = group_df["Angle Errors"].to_numpy(dtype=float).reshape((batch_size, N))
    omega_norms = group_df["Omega Errors"].to_numpy(dtype=float).reshape((batch_size, N))
    # Extract the "converged" trajs using the mean of the tail
    final_omegas = np.mean(omega_norms[:, -tail_length:], axis=1, keepdims=True)
    final_angles = np.mean(principal_rotation_angles[:, -tail_length:], axis=1, keepdims=True)
    
    # 3. Condition: Is the distance from the *average* final value within the neighborhood?
    cond = (np.abs(omega_norms - final_omegas) < omega_tol) & \
           (np.abs(principal_rotation_angles - final_angles) < angle_tol)
    
    # 4. Backward accumulation to find where it *stays* within the neighborhood
    mask = np.logical_and.accumulate(cond[:, ::-1], axis=1)[:, ::-1]
    
    # 5. Calculate indices and stability checks
    stability_start_idx = np.argmax(mask, axis=1)
    
    # We use the tail_length as our minimum required stability steps to ensure 
    # the sequence actually settled around this average for a meaningful duration.
    min_stable_steps = tail_length 
    ever_stable = np.sum(mask, axis=1) >= min_stable_steps
    
    slew_time = stability_start_idx * dt
    
    # 6. Apply NaNs for Seaborn plotting
    slew_time_arr = np.array(slew_time, dtype=float) 
    slew_time_arr[~ever_stable] = np.nan

    if labels is None:
      group_label = group
    else:
      group_label = labels[g]
    
    # Append to DataFrame structure
    df_group = pd.DataFrame({
      "Group": group_label,
      "Slew Time": slew_time_arr,
      "Final Quaternion Error": quat_costs[:, -1],
      "Final Angle Error": final_angles[:, -1],
      # "Final Angle Error": principal_rotation_angles[:, -1],
      "Final Omega Error": final_omegas[:, -1],
    })
    all_data.append(df_group)

  # Merge all groups into one DataFrame
  stats_df = pd.concat(all_data, ignore_index=True)
  
  return stats_df

def plot_traj(
    dt: ArrayLike,
    trajectory: NDArray,
    controls: NDArray | None,
    target_state: NDArray | None,
    labels_states: Sequence[str] | None,
    labels_controls: Sequence[str] | None,
    legend: str | Sequence[str] | None,
    title: str | None,
) -> list[Figure]:
  # Ensure batch dimension exists
  try:
      import matplotlib.pyplot as plt
      import seaborn as sns
  except ImportError as exc:
      raise ImportError(
          "plot_traj requires optional dependencies: pip install 'raddo[extras]'"
      ) from exc

  trajectory = np.asarray(trajectory)
  if controls is not None:
    controls = np.asarray(controls)
  if target_state is not None:
    target_state = np.asarray(target_state)

  if trajectory.ndim < 3:
      trajectory = np.expand_dims(trajectory, 0)
  n_traj = trajectory.shape[0]

  if target_state is not None:
    if target_state.ndim < 2:
      target_state = np.tile(target_state,(trajectory.shape[1],1))
    
  # Optional controls
  if controls is not None and controls.ndim < 3:
      controls = np.expand_dims(controls, 0)
    
  if controls is not None and controls.shape[0] != n_traj:
      if controls.shape[0] != 1:
          raise ValueError(
              f"controls has batch size {controls.shape[0]}, expected 1 or {n_traj}"
          )
      controls = np.broadcast_to(controls, (n_traj, *controls.shape[1:]))
  
  # Colors for trajectories
  # colors = plt.cm.tab10(np.linspace(0.0, 1.0, n_traj))
  colors = sns.color_palette("deep", n_colors=max(n_traj, 1))
  
  num_states = trajectory.shape[2]
  num_controls = controls.shape[2] if controls is not None else 0
  
  fig, axes = plt.subplots(nrows=int(num_states + num_controls), figsize=(8, 12))
  
  # Normalize axes to a list
  try:
      iter(axes)
      axes_list = list(axes)
  except TypeError:
      axes_list = [axes]
  
  for i, ax in enumerate(axes_list):
      if i < num_states:
          for j in range(n_traj):
            if legend is None:
              ax.plot(trajectory[j, :, i], color=colors[j])
            else:
              ax.plot(trajectory[j, :, i], color=colors[j],label=legend[j])
          if target_state is not None:
            ax.plot(target_state[:,i], linestyle='--',color='k')
          if labels_states is not None and i < len(labels_states):
              ax.set_ylabel(labels_states[i])
      else:
          ci = i - num_states
          if controls is not None:
              for j in range(n_traj):
                if legend is None:
                  ax.plot(controls[j, :, ci], color=colors[j])
                else:
                  ax.plot(controls[j, :, ci], color=colors[j], label=legend[j])
              if labels_controls is not None and ci < len(labels_controls):
                  ax.set_ylabel(labels_controls[ci])
      ax.grid(True)
      if legend is not None:
        ax.legend()
  if title is not None:
    axes_list[0].set_title(title)
   
  return [fig]

def plot_costs(
    df: DataFrame,
    batch_size: int,
    batch_idx: int | Sequence[int] | NDArray | None,
    N: int,
    title: str | None,
    legend: str | Sequence[str] | None,
    angle_threshold: float,
    omega_threshold: float,
    time_max: float | None,
    plot_stats: bool,
    compare_groups: bool,
    filename: str | None,
) -> tuple[DataFrame, list[Figure]]:
    """
    Plots aggregate statistics (Max, 95th, 50th, Mean) or all costs for all trajectories
    Supports trajectory shapes: [N, nx] or [batch, N, nx]
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise ImportError(
            "plot_costs requires optional dependencies: pip install 'raddo[extras]'"
        ) from exc

    groups = df["Group"].unique()
    n_groups = len(groups)
    figs: list[Figure] = []

    # Allow lists/tuples for batch_idx (numpy fancy indexing needs an array, not a tuple)
    if batch_idx is not None and not isinstance(batch_idx, (int, np.integer)):
        batch_idx = np.asarray(batch_idx)

    metric_configs = [
        {"key": "Angle Errors", "threshold": angle_threshold, "ylabel": "Angle (deg)", "suffix": "_angle"},
        {"key": "Omega Errors", "threshold": omega_threshold, "ylabel": r"$\|\omega\|_2$ (deg/s)", "suffix": "_omega"}
    ]

    if compare_groups:
        # Assign a distinct color to each group
        group_colors = sns.color_palette("tab10", n_colors=n_groups)

        for m in metric_configs:
            fig, a = plt.subplots(figsize=(8, 4), dpi=150)
            figs.append(fig)

            for g_idx, group in enumerate(groups):
                group_df = df.loc[df["Group"] == group]
                time_vector = group_df["Time"].to_numpy(dtype=float).reshape((batch_size, N))[0]
                data = group_df[m["key"]].to_numpy(dtype=float).reshape((batch_size, N))

                if batch_idx is not None:
                    data = data[batch_idx].reshape(-1, N)

                batch_select = data.shape[0]
                g_color = group_colors[g_idx]

                if plot_stats:
                    # Plot only the mean of each group
                    avg = np.mean(data, axis=0)
                    a.plot(time_vector, avg, color=g_color, linewidth=2.5, label=str(group))
                else:
                    # Plot all trajectories for the group with the same color
                    for b in range(batch_select):
                        lbl = str(group) if b == 0 else None  # Label once to prevent duplicate legend entries
                        a.plot(time_vector, data[b], color=g_color, linewidth=1.5, alpha=0.7, label=lbl)

            # # Threshold line
            # if m["threshold"] is not None:
            #     a.axhline(y=m["threshold"], color="black", linestyle="--", label=r"$\text{threshold}$")

            # Styling and Housekeeping
            a.set_ylabel(m["ylabel"])
            a.set_xlabel("Time (seconds)")

            if title:
                a.set_title(title, fontsize=14, fontweight='bold', y=1.02)

            a.grid(True, linestyle="-", alpha=0.15)
            sns.despine(ax=a)

            if time_max is None:
                a.set_xlim([time_vector[0], time_vector[-1]])
            else:
                a.set_xlim([time_vector[0], time_max])
            a.set_ylim(bottom=0)

            # Legend
            handles, labels = a.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if len(by_label) > 0:
                a.legend(by_label.values(), by_label.keys(), loc="upper right", frameon=True, framealpha=0.95, edgecolor="black")

            plt.tight_layout()
            if filename is not None:
              fig_dir = Path("figures")
              fig_dir.mkdir(parents=True, exist_ok=True)
              
              fname = f"costs_costs_compared{m['suffix']}_{filename}"
              file_path = fig_dir / fname
              plt.savefig(file_path)
             

    else:
        # Plot each group on their own plot
        for g in range(n_groups):
            group = df["Group"].unique()[g]
            group_df = df.loc[df["Group"] == group]
            time_vector = group_df["Time"].to_numpy(dtype=float).reshape((batch_size, N))
            time_vector = time_vector[0]
            quat_costs = group_df["Quaternion Errors"].to_numpy(dtype=float).reshape((batch_size, N))
            angle_costs = group_df["Angle Errors"].to_numpy(dtype=float).reshape((batch_size, N))
            omega_costs = group_df["Omega Errors"].to_numpy(dtype=float).reshape((batch_size, N))
            if batch_idx is not None:
              quat_costs = quat_costs[batch_idx].reshape(-1, N)
              angle_costs = angle_costs[batch_idx].reshape(-1, N)
              omega_costs = omega_costs[batch_idx].reshape(-1, N)
            batch_select = quat_costs.shape[0]
              
          
            # Define shared labels and configurations to decouple the loop
            metrics_info = [
                {"data": angle_costs, "threshold": angle_threshold, "ylabel": "Angle (deg)", "suffix": "_angle"},
                {"data": omega_costs, "threshold": omega_threshold, "ylabel": r"$\|\omega\|_2$ (deg/s)", "suffix": "_omega"}
            ]
    
            for idx, m in enumerate(metrics_info):
                # Create a completely separate figure for each metric
                fig, a = plt.subplots(figsize=(8, 4), dpi=150)
                figs.append(fig)
                
                if plot_stats:
                    blues_palette = sns.light_palette("steelblue", n_colors=4, reverse=False)
                    c_max = blues_palette[1]  
                    c_95  = blues_palette[2]  
                    c_50  = blues_palette[3]  
                    c_avg = "red"             
                    
                    data = m["data"]
                    mx = np.max(data, axis=0)
                    p95 = np.percentile(data, 95, axis=0)
                    p50 = np.percentile(data, 50, axis=0)
                    avg = np.mean(data, axis=0)
                    
                    a.fill_between(time_vector, 0, mx, color=c_max, alpha=0.3, label="Max error")
                    a.fill_between(time_vector, 0, p95, color=c_95, alpha=0.4, label="95th percentile")
                    a.fill_between(time_vector, 0, p50, color=c_50, alpha=0.2, label="50th percentile")
                    
                    a.plot(time_vector, mx, color=c_max, linewidth=1.5)
                    a.plot(time_vector, p95, color=c_95, linewidth=2.0)
                    a.plot(time_vector, p50, color=c_50, linewidth=2.5)
                    a.plot(time_vector, avg, color=c_avg, linewidth=2.5, label="Average")
                    
                    a.axhline(y=m["threshold"], color="black", linestyle="--", label=r"$\text{threshold}$")
                    
                else:
                    colors = sns.color_palette("deep", n_colors=max(batch_select, 1))
                    for b in range(batch_select):
                        lbl = legend[b] if isinstance(legend, (list, tuple, np.ndarray)) and b < len(legend) else (legend if b == 0 else None)
                        a.plot(time_vector, m["data"][b], label=lbl, color=colors[b], linewidth=2)
            
                # Styling and housekeeping per independent plot
                a.set_ylabel(m["ylabel"])
                a.set_xlabel("Time (seconds)")
            
                if title:
                    a.set_title(title, fontsize=14, fontweight='bold', y=1.02)
                  
                a.grid(True, linestyle="-", alpha=0.15)
                sns.despine(ax=a)
                if time_max is None:
                  a.set_xlim([time_vector[0], time_vector[-1]])
                else:
                  a.set_xlim([time_vector[0], time_max])
                a.set_ylim(bottom=0)
                
                handles, labels = a.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                if len(by_label) > 0:
                    a.legend(by_label.values(), by_label.keys(), loc="upper right", frameon=True, framealpha=0.95, edgecolor="black")
                
                plt.tight_layout()
                if filename is not None:
                  fig_dir = Path("figures")
                  fig_dir.mkdir(parents=True, exist_ok=True)
                  
                  fname = f"costs_{group.lower()}{m['suffix']}_{filename}"
                  file_path = fig_dir / fname
                  plt.savefig(file_path)
                 

    return df, figs

def plot_kde(
    stats_df: DataFrame,
    angle_threshold: float,
    omega_threshold: float,
    time_hist_max: float | None,
    angle_hist_max: float | None,
    omega_hist_max: float | None,
    legend: str | Sequence[str] | None,
    title: str | None,
    verbose: bool,
    filename: str | None,
) -> tuple[DataFrame, list[Figure]]:
    """
    Plots KDE distributions for Slew Time, Final Quat Error, and Final Omega Error as individual figures.
    Supports single trajectory arrays or a stacked array of multiple groups.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise ImportError(
            "plot_kde requires optional dependencies: pip install 'raddo[extras]'"
        ) from exc

    n_groups = stats_df["Group"].nunique()
    kde_kwargs = {"hue": "Group", "fill": True, "common_norm": False, "warn_singular": False, "alpha": 0.5}
    
    # --- Figure 1: Slew Time ---
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    sns.kdeplot(data=stats_df, x="Slew Time", ax=ax1, legend=False, clip=(0, time_hist_max), **kde_kwargs)
    ax1.set_xlabel("Time (s)")
    #ax1.set_title(f"{title}: Slew Time" if title else "Slew Time")
    ax1.set_xlim(0, time_hist_max)
    plt.tight_layout()
    if filename is not None:
        fig_dir = Path("figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        fname = f"kde_slew_time_{filename}"
        file_path = fig_dir / fname
        plt.savefig(file_path)
     

    # --- Figure 2: Final Principal Angle Error ---
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.kdeplot(data=stats_df, x="Final Angle Error", ax=ax2, legend=False, clip=(0, angle_hist_max), **kde_kwargs)
    #ax2.set_title("Final Principal Angle Error")
    ax2.set_xlabel(r"$\gamma_e$ (deg)")
    ax2.set_xlim(0, angle_hist_max)
    plt.tight_layout()
    if filename is not None:
        fig_dir = Path("figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        fname = f"kde_angle_error_{filename}"
        file_path = fig_dir / fname
        plt.savefig(file_path)
     

    # --- Figure 3: Final Omega Error ---
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    sns.kdeplot(data=stats_df, x="Final Omega Error", ax=ax3, clip=(0, omega_hist_max), **kde_kwargs)
    #ax3.set_title("Final Angular Velocity Norm")
    ax3.set_xlabel(r"$||\omega||$ (deg/s)")
    ax3.set_xlim(0, omega_hist_max)
    plt.tight_layout()
    if filename is not None:
        fig_dir = Path("figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        fname = f"kde_omega_error_{filename}"
        file_path = fig_dir / fname
        plt.savefig(file_path)
     

    if verbose:
        for g in range(n_groups):
            group_label = stats_df["Group"].unique()[g]
            group_data = stats_df.loc[stats_df["Group"] == group_label]
            final_angles = group_data["Final Angle Error"].to_numpy(dtype=float)
            final_omegas = group_data["Final Omega Error"].to_numpy(dtype=float)
            slew_time = group_data["Slew Time"].to_numpy(dtype=float)
            stable_trajs = ~np.isnan(slew_time)

            successful_trajs = ((final_angles < angle_threshold) & (final_omegas < omega_threshold) & (stable_trajs))
            stable_rate = np.mean(stable_trajs)*100
            success_rate = np.mean(successful_trajs) * 100
            print(f"--- {group_label} ---")
            print(f"Success Rate: {success_rate:.2f}%")
            print(f"Stable Rate: {stable_rate:.2f}%")
            print(f"Mean Converged Angle Error: {np.mean(final_angles):.3f} deg")
            print(f"Mean Final Angular Velocity: {np.mean(final_omegas):.3f} deg/s")

    return stats_df, [fig1, fig2, fig3]

def plot_violin_and_bar(
    stats_df_list: Sequence[DataFrame],
    angle_threshold: float,
    omega_threshold: float,
    time_hist_max: float | None,
    angle_hist_max: float | None,
    omega_hist_max: float | None,
    dataset_labels: Sequence[str] | None,
    title: str | None,
    verbose: bool,
    plot_bar_by_group: bool,
    filename: str | None,
) -> tuple[DataFrame, list[Figure]]:
    """
    Plots Violin distributions for Slew Time, Final Quat Error, and Final Omega Error as individual figures.
    Supports single trajectory arrays or a stacked array of multiple groups.
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError as exc:
        raise ImportError(
            "plot_violin_and_bar requires optional dependencies: pip install 'raddo[extras]'"
        ) from exc

    n_dfs = len(stats_df_list)
    df_combined = pd.concat(stats_df_list, keys=dataset_labels)
    stats_df = df_combined.reset_index()
    stats_df = stats_df.rename(columns={'level_0': 'Dataset'}).drop(columns=['level_1'], errors='ignore')
      
    n_groups = stats_df["Group"].nunique()

    if n_dfs > 1:
        violin_kwargs = {"x": "Group", "hue": "Dataset", "cut": 0, "legend": True}#, "inner": "quartile"} 
    else:
        violin_kwargs = {"x": "Group", "hue": "Group", "cut": 0, "legend": False}#, "inner": "quartile"}

    # Helper function to reduce styling redundancies across independent violins
    def finalize_violin_axes(ax: Axes) -> None:
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        if n_groups == 1:
            ax.set_xticks([])

    # --- Figure 1: Slew Time ---
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    sns.violinplot(data=stats_df, y="Slew Time", ax=ax1, **violin_kwargs)
    ax1.set_ylabel("Time (s)")
    ax1.set_xlabel("")
    ax1.set_ylim(0, time_hist_max)
    #ax1.set_title(f"{title}: Slew Time" if title else "Slew Time")
    if n_dfs > 1: ax1.get_legend().set_title("")
    finalize_violin_axes(ax1)
    plt.tight_layout()
    if filename is not None: 
      fig_dir = Path("figures")
      fig_dir.mkdir(parents=True, exist_ok=True)
      
      fname = f"violin_slew_{filename}"
      file_path = fig_dir / fname
      plt.savefig(file_path)
     

    # --- Figure 2: Final Principal Angle Error ---
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.violinplot(data=stats_df, y="Final Angle Error", ax=ax2, **violin_kwargs)
    #ax2.set_title("Final Principal Angle Error")
    ax2.set_ylabel(r"$\gamma_e$ (deg)")
    ax2.set_xlabel("")
    ax2.set_ylim(0, angle_hist_max)
    if n_dfs > 1: ax2.get_legend().set_title("")
    finalize_violin_axes(ax2)
    plt.tight_layout()
    if filename is not None: 
      fig_dir = Path("figures")
      fig_dir.mkdir(parents=True, exist_ok=True)
      
      fname = f"violin_angle_{filename}"
      file_path = fig_dir / fname
      plt.savefig(file_path)
     

    # --- Figure 3: Final Omega Error ---
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    sns.violinplot(data=stats_df, y="Final Omega Error", ax=ax3, **violin_kwargs)
    #ax3.set_title("Final Angular Velocity Norm")
    ax3.set_ylabel(r"$||\omega||$ (deg/s)")
    ax3.set_xlabel("")
    ax3.set_ylim(0, omega_hist_max)
    if n_dfs > 1: ax3.get_legend().set_title("")
    finalize_violin_axes(ax3)
    plt.tight_layout()
    if filename is not None: 
      fig_dir = Path("figures")
      fig_dir.mkdir(parents=True, exist_ok=True)
      
      fname = f"violin_omega_{filename}"
      file_path = fig_dir / fname
      plt.savefig(file_path)
     

    # --- Collect Data for Success Rate & Mean Plots ---
    success_data = []
    grouped = stats_df.groupby(["Dataset", "Group"], sort=False)
    for (dataset_label, group_label), group_data in grouped:
        final_angles = group_data["Final Angle Error"].to_numpy(dtype=float)
        final_omegas = group_data["Final Omega Error"].to_numpy(dtype=float)
        slew_time    = group_data["Slew Time"].to_numpy(dtype=float)
        stable_trajs = ~np.isnan(slew_time)

        successful_trajs = ((final_angles < angle_threshold) & (final_omegas < omega_threshold) & (stable_trajs))
        stable_rate = np.mean(stable_trajs) * 100
        success_rate = np.mean(successful_trajs) * 100
        
        mean_angle_error = np.mean(final_angles)
        mean_omega_error = np.mean(final_omegas)
        
        # Store metrics for plotting
        success_data.append({
            "Dataset": dataset_label, 
            "Group": group_label, 
            "Success Rate": float(success_rate),
            "Mean Angle Error": float(mean_angle_error),
            "Mean Omega Error": float(mean_omega_error)
        })

        if verbose:
          print(f" ---- {dataset_label} | {group_label} ---- ")
          print(f"Success Rate: {success_rate:.2f}%")
          print(f"Stable Rate: {stable_rate:.2f}%")
          print(f"Mean Converged Angle Error: {mean_angle_error:.3f} deg")
          print(f"Mean Final Angular Velocity: {mean_omega_error:.3f} deg/s")
          print()

    # --- Convert collected stats to DataFrame ---
    success_df = pd.DataFrame(success_data)
    bar_kwargs = {"x": "Group", "hue": "Dataset" if n_dfs > 1 else "Group", "legend": n_dfs > 1}

    # --- Figure 4: Success Rate Bar Plot ---
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.barplot(data=success_df, y="Success Rate", ax=ax4, **bar_kwargs)
    ax4.set_ylabel("Success Rate (%)")
    ax4.set_xlabel("")
    ax4.set_ylim(0, 100)
    if n_dfs > 1: ax4.get_legend().set_title("")
    finalize_violin_axes(ax4)
    plt.tight_layout()
    if filename is not None: 
      fig_dir = Path("figures")
      fig_dir.mkdir(parents=True, exist_ok=True)
      
      fname = f"bar_success_{filename}"
      file_path = fig_dir / fname
      plt.savefig(file_path)
     

    group_figs: list[Figure] = []
    if plot_bar_by_group:
      # --- Figure 5: Success Rate Bar Plot by Group ---
      # --- Individual Group Success Rate Bar Plots ---
      for target_group in success_df["Group"].unique():
          group_df = success_df[success_df["Group"] == target_group]
          
          fig_g, ax_g = plt.subplots(figsize=(10, 5))
          group_figs.append(fig_g)
          
          # Keep y and hue the same; x is just the single group
          sns.barplot(
              data=group_df, 
              x="Group", 
              y="Success Rate", 
              hue="Dataset" if n_dfs > 1 else "Group", 
              ax=ax_g
          )
          
          ax_g.set_ylabel("Success Rate (%)")
          ax_g.set_xlabel("")
          ax_g.set_xticks([]) # Removes the single x-tick label for a cleaner look
          ax_g.set_ylim(0, 100)
          #ax_g.set_title(f"Success Rate: {target_group}")
          
          if n_dfs > 1: 
              ax_g.get_legend().set_title("")
              
          finalize_violin_axes(ax_g)
          plt.tight_layout()
          
          if filename is not None: 
            fig_dir = Path("figures")
            fig_dir.mkdir(parents=True, exist_ok=True)
            
            fname = f"bar_success_{target_group}_{filename}"
            file_path = fig_dir / fname
            plt.savefig(file_path)

    return stats_df, [fig1, fig2, fig3, fig4, *group_figs]

def plot_3D(trajectory: NDArray) -> list[Figure]:
  """
  Plot 3D trajectories.
  
  Parameters:
  -----------
  trajectory : np.ndarray
      Trajectory data of shape (batch_size, N, 13) or (N, 13)
      where the first 3 states are position (x, y, z).
  """
  try:
      import matplotlib.pyplot as plt
  except ImportError as exc:
      raise ImportError(
          "plot_3D requires optional dependencies: pip install 'raddo[extras]'"
      ) from exc

  trajectory = np.asarray(trajectory)

  # Handle single trajectory case (N, 13) -> (1, N, 13)
  if (trajectory.ndim < 3):
      trajectory = np.expand_dims(trajectory, 0)
  n_traj = trajectory.shape[0]
  
  # Extract positions (first 3 states)
  positions = trajectory[:, :, :3]  # Shape: (batch_size, N, 3)
  
  # Create 3D figure
  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection='3d')
  
  
  # Color map for multiple trajectories
  colors = plt.cm.tab10(np.linspace(0, 1, n_traj))
  
  # Plot each trajectory
  for i in range(n_traj):
      x = positions[i, :, 0]
      y = positions[i, :, 1]
      z = positions[i, :, 2]
      
      # Plot the orbit line
      ax.plot(x, y, z, color=colors[i], label=f'Trajectory {i+1}', linewidth=1.5)
      
      # Mark start point (green) and end point (red)
      ax.scatter(x[0], y[0], z[0], color='green', s=50, marker='o', edgecolors='black')
      ax.scatter(x[-1], y[-1], z[-1], color='red', s=50, marker='s', edgecolors='black')
  
  # Plot origin (central body)
  #ax.scatter(0, 0, 0, color='yellow', s=50, marker='o', edgecolors='orange', label='Central Body')
  
  # Labels and title
  ax.set_xlabel('X [km]')
  ax.set_ylabel('Y [km]')
  ax.set_zlabel('Z [km]')
  ax.set_title('Orbit (ECI coords)')
  
  # Add legend if multiple trajectories
  # if n_traj > 1:
  #     ax.legend()
  
  # Equal aspect ratio
  max_range = float(np.max(np.abs(positions))) * 1.1
  ax.set_xlim([-max_range, max_range])
  ax.set_ylim([-max_range, max_range])
  ax.set_zlim([-max_range, max_range])
  
  plt.tight_layout()
   
  return [fig]
