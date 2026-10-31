from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from jax.typing import ArrayLike
    from pandas import DataFrame

    from raddo.controllers.base_controller import Controller
    from raddo.dynamics.base_dynamics import Dynamics

    from .learned_model_implementation import LearningApplicator


def sample_initial_states(
    batch_size: int,
    key: jax.Array,
    state_specs: Sequence[dict[str, Any]],
) -> jax.Array:
    """
    Generates a batch of random initial states based on provided specs.

    Args:
        batch_size: Number of states to sample.
        key: JAX key.
        state_specs: List of dicts, e.g., 
            [{'name': 'pos', 'shape': (3,), 'dist': 'uniform', 'min': -1, 'max': 1},
             {'name': 'rot', 'shape': (4,), 'dist': 'quaternion'}]
    """
    states = []
    
    for spec in state_specs:
        key, subkey = jrandom.split(key)
        dist_type = spec.get('dist', 'uniform')
        shape = (batch_size,) + spec.get('shape', (1,))

        if dist_type == 'uniform':
            val = jrandom.uniform(subkey, shape=shape, 
                                 minval=spec['min'], maxval=spec['max'])
        
        elif dist_type == 'normal':
            val = spec.get('mean', 0.0) + spec.get('std', 1.0) * jrandom.normal(subkey, shape=shape)

        elif dist_type == 'quaternion':
            # Specialized sampler for unit quaternions
            q = jrandom.normal(subkey, shape=shape)
            val = q / jnp.linalg.norm(q, axis=-1, keepdims=True)

        elif dist_type == 'constant':
            return jnp.broadcast_to(jnp.array(spec['value']), shape)

        states.append(val.reshape(batch_size, -1))
        #TODO: Potentially set trajectory up as pytrees (traj.pos instead of traj[:,0:3])
    return jnp.concatenate(states, axis=-1)

class TrajectoryGenerator(eqx.Module):
  dynamics: Dynamics
  dt: float

  def rk4_step(
      self,
      state: jax.Array,
      control: jax.Array,
      t: ArrayLike,
      external_dynamics_param: jax.Array,
      key: jax.Array,
      noise_std: ArrayLike,
  ) -> jax.Array:
    def dynamics_wrapper(s: jax.Array,k: jax.Array) -> jax.Array:
      key, noise_key = jrandom.split(k)
      u_noise = jrandom.normal(noise_key, shape=control.shape) * noise_std
      return self.dynamics.state_dot(s, control, t, external_dynamics_param, u_noise)

    # Note: quaternion projection is just identity unless defined in Dynamics class
    k1 = dynamics_wrapper(state,key)
    k2 = dynamics_wrapper(self.dynamics.quaternion_projection(state + 0.5 * self.dt * k1),key)
    k3 = dynamics_wrapper(self.dynamics.quaternion_projection(state + 0.5 * self.dt * k2),key)
    k4 = dynamics_wrapper(self.dynamics.quaternion_projection(state + self.dt * k3),key)
  
    dx = (self.dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # key1, key = jrandom.split(key)
    # noise = jrandom.normal(key, shape=state.shape) * noise_std * jnp.sqrt(self.dt)

    return self.dynamics.quaternion_projection(state + dx)

  def generate_trajectory(
        self,
        initial_state: jax.Array,
        key: jax.Array,
        target_state: jax.Array | None=None,
        noise_std: ArrayLike=0.0,
        num_steps: int=200,
        external_dynamics_params: jax.Array | None=None,
        external_dynamics_params_est: jax.Array | None=None,
        high_level_controller: Controller | None=None,
        replan_freq: int=10,
        low_level_controller: Controller | None=None,
        control_sequence: jax.Array | None=None,
        learning_applicator: LearningApplicator | None = None,
    ) -> tuple[jax.Array, jax.Array]:
    """
        Unified entry point, dispatches based on what's provided to generate trajectories for
        fixed control sequence, high-level control, low-level control, high and low-level, with/without adaptation

        Inputs:
          initial_state: initial state 
          key: noise key for jax.random
          target_state: goal state
          noise_std: noise standard deviation for noise of disturbance forces in dynamics
          num_steps: how long to propagate trajectory
          external_dynamics_params: parameters used in dynamics not propagated as part of state (e.g. orbital position, magnetic field values)
          external_dynamics_params_est: Estimates of dynamics paramss given to controller (with dynamics still propagated w/ true params) e.g. noisy magnetic field measurements
          high_level_controller: Controller - high-level controller that does planning and executes in MPC-like fashion
          replan_freq: how often high-level controller is called to replan
          low_level_controller: Controller - low-level tracking controller (e.g. LQR, PID)
          control_sequence: fixed control sequence to propagate
          learning_applicator: defines how we are using learning - could be fixed NN model, or with adaptation 

        Returns:
          trajectory: Shape (N + 1, nx)
          controls: Shape (N, nu)
    """
    # Trims and/or pads to the proper length based on num_steps and high_level_controller.horizon if given
    ext_dyn_params, ext_dyn_params_est = self._prepare_external_params(
        external_dynamics_params, external_dynamics_params_est,
        num_steps, high_level_controller
    )

    # Open-loop control sequence
    if control_sequence is not None:
        return self._generate_trajectory_seq(
            initial_state, target_state, key, noise_std,
            num_steps, ext_dyn_params, control_sequence
        )

    # Propagates dynamics with no control input
    if high_level_controller is None and low_level_controller is None:
        control_sequence = jnp.zeros((num_steps, self.dynamics.num_controls))
        return self._generate_trajectory_seq(
            initial_state, target_state, key, noise_std,
            num_steps, ext_dyn_params, control_sequence
        )

    # Expand target to trajectory-length reference for low-level control
    if low_level_controller is not None and high_level_controller is None:
        if target_state.ndim < 2:
            target_state = jnp.tile(target_state, (num_steps + 1, 1))

    # High-level or low-level control (with or without low-level tracking) (with or without online adaptation)
    return self._generate_trajectory_core(
        initial_state, target_state, key, noise_std, num_steps,
        ext_dyn_params, ext_dyn_params_est,
        high_level_controller, low_level_controller,
        replan_freq, learning_applicator
    )

  def generate_trajectory_batch(self, initial_states: jax.Array, key: jax.Array, batch_size: int,
                               target_states: jax.Array | None=None, noise_std: ArrayLike=0.0, num_steps: int=200,
                               external_dynamics_params: jax.Array | None=None,
                               external_dynamics_params_est: jax.Array | None=None,
                               high_level_controller: Controller | None=None, replan_freq: int=10,
                               low_level_controller: Controller | None=None, control_sequence: jax.Array | None=None,
                               learning_applicator: LearningApplicator | None=None, chunk_size: int=0) -> tuple[jax.Array, jax.Array]:

    keys = jrandom.split(key, batch_size)
    actual_chunk_size = chunk_size or batch_size

    def prepare_input(x: jax.Array | None, dummy_shape: tuple[int, ...]) -> jax.Array | None:
        if x is None:
            return None
        if x.ndim == len(dummy_shape) + 1:  # already batched
            return x
        return jnp.broadcast_to(x, (batch_size, *x.shape))

    b_init = prepare_input(initial_states, (self.dynamics.num_states,))
    b_target = prepare_input(target_states, (self.dynamics.num_states,))
    b_extern = prepare_input(external_dynamics_params, (num_steps, 1))
    b_extern_est = prepare_input(external_dynamics_params_est, (num_steps, 1))
    b_ctrl_seq = prepare_input(control_sequence, (num_steps, self.dynamics.num_controls))

    def map_fn(
        args: tuple[jax.Array, jax.Array | None, jax.Array, jax.Array | None, jax.Array | None, jax.Array | None],
    ) -> tuple[jax.Array, jax.Array]:
        init, target, k, extern, extern_est, c_seq = args
        return self.generate_trajectory(
            init, k, target_state=target, noise_std=noise_std,
            num_steps=num_steps, external_dynamics_params=extern,
            external_dynamics_params_est=extern_est,
            high_level_controller=high_level_controller,
            replan_freq=replan_freq, low_level_controller=low_level_controller,
            control_sequence=c_seq, learning_applicator=learning_applicator,
        )

    batched_args = (b_init, b_target, keys, b_extern, b_extern_est, b_ctrl_seq)

    if chunk_size and chunk_size < batch_size:
        return jax.lax.map(map_fn, batched_args, batch_size=chunk_size)
    else:
        return jax.vmap(map_fn)(batched_args)

  @eqx.filter_jit
  def _generate_trajectory_seq(
    self,
    initial_state: jax.Array, 
    target_state: jax.Array | None,
    key: jax.Array,
    noise_std: ArrayLike,
    num_steps: int,
    external_dynamics_params: jax.Array,
    control_sequence: jax.Array
    ) -> tuple[jax.Array, jax.Array]: 
    """
    Generate trajectory using  control sequence.
    
    Returns:
        trajectory: Shape (N + 1, nx)
        controls: Shape (N, nu)
    """  
    scan_inputs = (control_sequence, external_dynamics_params)
    
    def scan_step(
        carry: tuple[jax.Array, jax.Array, int | jax.Array],
        scan_input: tuple[jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array, int | jax.Array], tuple[jax.Array, jax.Array]]:
      state, key, i = carry
      control_input, extern_dyn_param = scan_input
      t_current = i * self.dt
      key, rk4_key = jrandom.split(key)
      next_state = self.rk4_step(state, control_input, t_current, extern_dyn_param, rk4_key, noise_std)
      return (next_state, key, i+1), (state, control_input)
  
    # Run scan
    init_carry = (initial_state, key, 0)
    (final_state, final_key, _), (trajectory, controls) = jax.lax.scan(scan_step, init_carry, scan_inputs)
  
    # Add final state to end of trajectory
    trajectory = jnp.vstack([trajectory, final_state[None, :]])
  
    return trajectory, controls

  @eqx.filter_jit
  def _generate_trajectory_core(
      self,
      initial_state: jax.Array, target_state: jax.Array | None, key: jax.Array, noise_std: ArrayLike, num_steps: int,
      external_dynamics_params: jax.Array,
      external_dynamics_params_est: jax.Array,
      high_level_controller: Controller,   # None -> no replanning
      low_level_controller: Controller,    # None -> use nominal_cntrl[0] directly from high-level cntrl output
      replan_freq: int,
      learning_applicator: LearningApplicator,
  ) -> tuple[jax.Array, jax.Array]:
      use_high_level = high_level_controller is not None
      use_low_level = low_level_controller is not None
      use_learning = learning_applicator is not None
  
      if use_high_level:
          horizon = high_level_controller.horizon
      else:
          # No replanning
          horizon = num_steps
  
      params_dim = external_dynamics_params.shape[1]
      est_params_dim = external_dynamics_params_est.shape[1]
      key, init_cntrl_key = jrandom.split(key)
  
      if use_high_level:
          init_nom_traj = jnp.tile(initial_state, (horizon + 1, 1))
          init_nom_cntrl = 0.001 * jrandom.normal(
              init_cntrl_key, shape=(horizon, self.dynamics.num_controls)
          )
      else:
          # Low-level only: the nominal trajectory is the target
          init_nom_traj = target_state  # (num_steps+1, nx)
          init_nom_cntrl = jnp.zeros((num_steps, self.dynamics.num_controls))

      # Controller param used in low-level control (set to 0 if low-level control is not being used)
      cntrl_param_low = (
          low_level_controller.cntrl_param_init if use_low_level
          else jnp.zeros(1)
      )
      learning_carry = learning_applicator.init_state() if use_learning else None
  
      def scan_step(
          carry: tuple[jax.Array, jax.Array, int | jax.Array, jax.Array, jax.Array, jax.Array, Any],
          _: None,
      ) -> tuple[tuple[jax.Array, jax.Array, int | jax.Array, jax.Array, jax.Array, jax.Array, Any], tuple[jax.Array, jax.Array]]:
          state, key, i, cntrl_param_low, nominal_traj, nominal_cntrl, learning_carry = carry
          t_current = i * self.dt
          key, rk4_key, cntrl_key, learning_key = jrandom.split(key, 4)
  
          # Slice true dynamics params
          dyn_params_slice = jax.lax.dynamic_slice(
              external_dynamics_params, (i, 0), (horizon, params_dim)
          )
          dyn_params_est_slice = jax.lax.dynamic_slice(
              external_dynamics_params_est, (i, 0), (horizon, est_params_dim)
          )
  
          # Use learned model (with or without adaptation) (e.g. NN magnetic field model uses orbital position to get magnetic field values)
          if use_learning:
              history_len = getattr(learning_applicator, 'history', 0) # default to 0 as placeholder
           
              dyn_params_est_history = jax.lax.dynamic_slice(
                  external_dynamics_params_est,
                  (jnp.maximum(i - history_len, 0), 0), # guarantees positive indices (no wrapping to future values), adapt conditional handles case where i < history
                  (history_len, est_params_dim)
              )
              controller_dyn_params, learning_carry = learning_applicator(learning_carry, dyn_params_est_slice, dyn_params_est_history, i, self.dt, horizon, learning_key)
          else:
              # if estimated (e.g. noisy) params aren't given, these are the true params (set in _prepare_external_params)
              controller_dyn_params = dyn_params_est_slice
  
          # High-level controller
          if use_high_level:
              def controller_wrapper(operand: tuple[Any, ...]) -> tuple[jax.Array, jax.Array]:
                  return high_level_controller(*operand)
  
              def no_update(operand: tuple[Any, ...]) -> tuple[jax.Array, jax.Array]:
                  *_, nominal_traj, nominal_cntrl = operand
                  # Shift nominal trajectory and nominal cntrl
                  nominal_traj = jnp.concatenate((nominal_traj[1:], nominal_traj[-1:]), axis=0)
                  nominal_cntrl = jnp.concatenate((nominal_cntrl[1:], nominal_cntrl[-1:]), axis=0)
                  return nominal_traj, nominal_cntrl
  
              nominal_traj, nominal_cntrl = jax.lax.cond(
                  i % replan_freq == 0,
                  controller_wrapper,
                  no_update,
                  operand=(state, target_state, cntrl_key, controller_dyn_params, nominal_traj, nominal_cntrl)
              )
          else:
              # Low-level only: shift the fixed reference each step
              nominal_traj = jnp.concatenate((nominal_traj[1:], nominal_traj[-1:]), axis=0)
              nominal_cntrl = jnp.concatenate((nominal_cntrl[1:], nominal_cntrl[-1:]), axis=0)
  
          # Apply tracking controller, or just directly return cntrl from high-level control
          if use_low_level:
              control_input, cntrl_param_low = low_level_controller(state, target_state, cntrl_key, dyn_params_slice, nominal_traj, nominal_cntrl, cntrl_param_low)
          else:
              control_input = nominal_cntrl[0]
  
          # Propagate one step dynamics
          next_state = self.rk4_step(
              state, control_input, t_current, dyn_params_slice[0],
              rk4_key, noise_std
          )
  
          new_carry = (next_state, key, i + 1, cntrl_param_low, nominal_traj, nominal_cntrl, learning_carry)
          return new_carry, (state, control_input)

      # Initialize and run scan for num_steps
      init_carry = (initial_state, key, 0, cntrl_param_low, init_nom_traj, init_nom_cntrl, learning_carry)
      (final_state, *_), (trajectory, controls) = jax.lax.scan(scan_step, init_carry, length=num_steps)

      # Concatenate final state 
      trajectory = jnp.vstack([trajectory, final_state[None, :]])
      return trajectory, controls

  def _prepare_external_params(
      self,
      external_dynamics_params: jax.Array,
      external_dynamics_params_est: jax.Array,
      num_steps: int,
      high_level_controller: Controller,
  ) -> tuple[jax.Array, jax.Array]:
      """
      Validate and pad/trim external dynamics params to the required length.
      
      The required length depends on whether a high-level controller is used:
      - No controller: num_steps
      - With controller: num_steps + horizon (controller needs to see ahead)
      
      Returns:
          ext_dyn_params: (required_length, params_dim)
          ext_dyn_params_est: (required_length, est_params_dim) or None
      """
      if high_level_controller is not None:
          horizon = high_level_controller.horizon
          required_length = num_steps + horizon
      else:
          required_length = num_steps
  
      # True params (for dynamics propagation)
      if external_dynamics_params is None:
          ext_dyn_params = jnp.zeros((required_length, 1))
      else:
          assert external_dynamics_params.shape[0] >= required_length, (
              f"external_dynamics_params must have at least {required_length} rows, "
              f"got {external_dynamics_params.shape[0]}"
          )
          ext_dyn_params = external_dynamics_params[:required_length]
  
      # Estimated params given to controller (e.g. noisy measurements)
      if external_dynamics_params_est is None:
          # Set to true params if not provided
          ext_dyn_params_est = ext_dyn_params
      else:
          assert external_dynamics_params_est.shape[0] >= required_length, (
              f"external_dynamics_params_est must have at least {required_length} rows, "
              f"got {external_dynamics_params_est.shape[0]}"
          )
          ext_dyn_params_est = external_dynamics_params_est[:required_length]
  
      return ext_dyn_params, ext_dyn_params_est

  @eqx.filter_jit
  def _generate_trajectory_batch_seq(self,
                                initial_states: jax.Array,
                                target_states: jax.Array | None,
                                key: jax.Array,
                                batch_size: int,
                                noise_std: ArrayLike,
                                num_steps: int,
                                external_dynamics_params: jax.Array,
                                control_sequence: jax.Array) -> tuple[jax.Array, jax.Array]:
    """ Batch trajectory generator without conditionals (for computational graph simplicity for jit) for multiple control sequence inputs (for use in MPPI or CEM)"""
    keys = jrandom.split(key, batch_size)
    batch_fn = jax.vmap(self._generate_trajectory_seq, in_axes=(None, None, 0, None, None, None, 0))
    return batch_fn(initial_states, target_states, keys, noise_std, num_steps, external_dynamics_params, control_sequence)

  def plot_traj(
      self,
      trajectory: jax.Array,
      controls: jax.Array | None=None,
      target_state: jax.Array | None=None,
      labels_states: Sequence[str] | None=None,
      labels_controls: Sequence[str] | None=None,
      legend: str | Sequence[str] | None=None,
      title: str | None=None,
  ) -> None:
    """
    Plot one or more trajectories, and optionally controls.
    Each trajectory is shaped (N+1, nx).
    Controls, if provided, are shaped (N, nu).
    labels_states and labels_controls optionally override y-axis labels.
    Defaults use self.dynamics.names_states and self.dynamics.names_controls.
    """
    from .plotting import plot_traj

    # Labels (defaults to dynamics names)
    labels_states = labels_states if labels_states is not None else self.dynamics.names_states
    labels_controls = labels_controls if labels_controls is not None else (
        self.dynamics.names_controls if controls is not None else None)
    dt = self.dt
    
    return plot_traj(dt, trajectory, controls, target_state, labels_states, labels_controls, legend, title)

  def plot_costs(
      self,
      trajectory: jax.Array | None=None,
      target_state: jax.Array | None=None,
      df: DataFrame | None=None,
      batch_idx: int | Sequence[int] | jax.Array | None=None,
      angle_threshold: float=15.0,
      omega_threshold: float=5.0,
      time_max: float | None=None,
      title: str | None=None,
      legend: str | Sequence[str] | None=None,
      plot_stats: bool=False,
      compare_groups: bool=False,
      filename: str | None=None,
  ) -> DataFrame:
    """
    Plots quaternion and angular velocity costs over the horizon.
    Supports trajectory shapes: [N, nx] or [batch, N, nx]
    """
    from .plotting import plot_costs
    from .save_df import save_data_to_pd_df

    quat_start = self.dynamics.params.get("quat_start")
    dt = self.dt

    if df is None:
      df = save_data_to_pd_df(trajectory, target_state, filename=None, labels=legend, dt=dt, quat_start=quat_start)
      if trajectory.ndim == 2:
          trajectory = trajectory[None, None, :, :]
      elif trajectory.ndim == 3:
          trajectory = trajectory[None, :, :, :]
      
      batch_size = trajectory.shape[1]
      N = trajectory.shape[2]
    else:
      batch_size = int(df["Time"].value_counts()[0] / df["Group"].nunique())
      N = df["Time"].nunique()

    return plot_costs(df=df, batch_size=batch_size, batch_idx=batch_idx, N=N, title=title, legend=legend, angle_threshold=angle_threshold, omega_threshold=omega_threshold, time_max=time_max, plot_stats=plot_stats,compare_groups=compare_groups,filename=filename)
  

  def plot_kde(
      self,
      trajectories: jax.Array | None=None,
      target_states: jax.Array | None=None,
      df: DataFrame | None=None,
      angle_threshold: float=15.0,
      omega_threshold: float=5.0,
      angle_stability_tol: float=5.0,
      omega_stability_tol: float=5.0,
      tail_length: int=100,
      time_hist_max: float | None=None,
      angle_hist_max: float | None=None,
      omega_hist_max: float | None=None,
      legend: str | Sequence[str] | None=None,
      title: str | None=None,
      verbose: bool=False,
      filename: str | None=None,
  ) -> None:
    """
    Plots kde desnity plots for Slew Time, Final Quat Error, and Final Omega Error.
    """
    from .plotting import create_stats_df, plot_kde
    from .save_df import save_data_to_pd_df

    dt = self.dt
    quat_start = self.dynamics.params.get("quat_start")

    if df is None:
      df = save_data_to_pd_df(trajectories, target_states, filename=None, labels=legend, dt=dt, quat_start=quat_start)
      if trajectories.ndim == 2:
          trajectories = trajectories[None, None, :, :]
      elif trajectories.ndim == 3:
          trajectories = trajectories[None, :, :, :]
      
      batch_size = trajectories.shape[1]
      N = trajectories.shape[2]
    else:
      batch_size = int(df["Time"].value_counts()[0] / df["Group"].nunique())
      N = df["Time"].nunique()

    stats_df = create_stats_df(df, batch_size, N, angle_stability_tol, omega_stability_tol, tail_length, labels=None)

    if time_hist_max is None:
        time_hist_max = df["Time"].max()
    if angle_hist_max is None:
        angle_hist_max = angle_threshold
    if omega_hist_max is None:
        omega_hist_max = omega_threshold

    return plot_kde(stats_df, angle_threshold, omega_threshold, time_hist_max, angle_hist_max, omega_hist_max, legend, title, verbose, filename)

  def plot_violin_and_bar(
      self,
      trajectories: jax.Array | None=None,
      target_states: jax.Array | None=None,
      df_list: Sequence[DataFrame] | None=None,
      angle_threshold: float=15.0,
      omega_threshold: float=5.0,
      angle_stability_tol: float=5.0,
      omega_stability_tol: float=5.0,
      tail_length: int=100,
      time_hist_max: float | None=None,
      angle_hist_max: float | None=None,
      omega_hist_max: float | None=None,
      legend: str | Sequence[str] | None=None,
      dataset_labels: Sequence[str] | None=None,
      title: str | None=None,
      verbose: bool=False,
      plot_bar_by_group: bool=False,
      filename: str | None=None,
  ) -> None:
    """
    Plots histograms for Slew Time, Final Quat Error, and Final Omega Error.
    """
    from .plotting import create_stats_df, plot_violin_and_bar
    from .save_df import save_data_to_pd_df

    dt = self.dt
    quat_start = self.dynamics.params.get("quat_start")

    if df_list is None:
      df = save_data_to_pd_df(trajectories, target_states, filename=None, labels=legend, dt=dt, quat_start=quat_start)
      df_list = [df]
      if trajectories.ndim == 2:
          trajectories = trajectories[None, None, :, :]
      elif trajectories.ndim == 3:
          trajectories = trajectories[None, :, :, :]
      
      batch_size = trajectories.shape[1]
      N = trajectories.shape[2]

    else:
      df = df_list[0]
      batch_size = int(df["Time"].value_counts()[0] / df["Group"].nunique())
      N = df["Time"].nunique()

    stats_df_list = []
    if dataset_labels is None:
      datasets = []
    else:
      datasets = dataset_labels
      
    for i in range(len(df_list)):
      if dataset_labels is None:
        datasets.append(f"Dataset_{i+1}")
      stats_df_list.append(create_stats_df(df_list[i], batch_size, N, angle_stability_tol, omega_stability_tol, tail_length, labels=None))

    if time_hist_max is None:
        time_hist_max = df_list[0]["Time"].max()
    if angle_hist_max is None:
        angle_hist_max = angle_threshold
    if omega_hist_max is None:
        omega_hist_max = omega_threshold

    return plot_violin_and_bar(stats_df_list, angle_threshold, omega_threshold, time_hist_max, angle_hist_max, omega_hist_max, datasets, title, verbose, plot_bar_by_group, filename)


  def plot_3D(self, trajectory: jax.Array) -> None:
    """
    Plot 3D trajectories.
    
    Parameters:
    -----------
    trajectory : jax.Array
        Trajectory data of shape (batch_size, N, 13) or (N, 13)
        where the first 3 states are position (x, y, z).
    """
    from .plotting import plot_3D

    return plot_3D(trajectory)


