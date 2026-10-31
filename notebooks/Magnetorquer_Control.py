import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    # JAX pre-allocates memory when it is imported, this stops that so memory is only used as needed
    import os
    # Prevent memory pre-allocation for flexible memory management (useful if working on shared GPU)
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9' # Use 90% of GPU memory default is '.75'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


    import time

    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    import marimo as mo
    import pandas as pd
    import seaborn as sns

    from raddo.controllers import FeedbackController, QPController
    from raddo.dynamics import (
        Earth,
        Uranus,
        OrbitDynamics,
        MagnetorquerDynamics,
        SpacecraftDynamicsTorque,
    )
    from raddo.utils import (
        AdaptMagModel,
        LearnedPosToMagMap,
        TrajectoryGenerator,
        coord,
        last_layer_update,
        load_model,
        sample_initial_states,
        save_data_to_pd_df,
    )

    # Jax configuration
    jax.config.update("jax_default_matmul_precision", "highest") # required for vmap to work properly, weird bug
    jax.config.update("jax_enable_x64", True) # enables 64-bit precision

    # Seaborn plotting style
    sns.set_theme(context='paper', style='whitegrid', font='serif', font_scale=1.5)


@app.cell
def _():
    jax.devices()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Initialize simulation environment, dynamics, and controllers
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Set base parameters, and select learned magnetic field models to use (generated in ```B_field_Learning.py```)
    """)
    return


@app.cell
def _():
    # Define parameters that must be consistent for orbit and spacecraft attitude dynamics
    dt = 0.1
    planet = Uranus # Earth, Uranus
    return dt, planet


@app.cell
def _(planet):
    import urllib.request
    import tempfile

    def _download_to_tempfile(url: str, suffix: str = ".eqx") -> str:
        """Download a remote file and save it locally, returning the local path."""
        data = urllib.request.urlopen(url).read()
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return path

    _eqx_url_base = "https://raw.githubusercontent.com/JHU-ACEL/uranus-mpc-data/main/models"


    # Load learned magnetic field models
    if planet is Earth:
        _model_name = "earth_b_4d.eqx"
        _eqx_url = f"{_eqx_url_base}/{_model_name}"
        _local_path = _download_to_tempfile(_eqx_url)
        model_s, _ = load_model(filename=_local_path)

        _model_name = "earth_b_4d_coarse.eqx"
        _eqx_url = f"{_eqx_url_base}/{_model_name}"
        _local_path = _download_to_tempfile(_eqx_url)
        model_coarse, _ = load_model(filename=_local_path)
    elif planet is Uranus:
        _model_name = "uranus_b_4d.eqx"
        _eqx_url = f"{_eqx_url_base}/{_model_name}"
        _local_path = _download_to_tempfile(_eqx_url)
        model_s, _ = load_model(filename=_local_path)

        _model_name = "uranus_b_4d_coarse.eqx"
        _eqx_url = f"{_eqx_url_base}/{_model_name}"
        _local_path = _download_to_tempfile(_eqx_url)
        model_coarse, _ = load_model(filename=_local_path)

    os.remove(_local_path)
    return model_coarse, model_s


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Generate orbital trajectories and magnetic field data
    """)
    return


@app.cell
def _(dt, planet):
    # ==================== Orbit Dynamics =================== #
    N_orbit = 5000
    orbit_dynamics = OrbitDynamics(Planet=planet)
    orbit_system = TrajectoryGenerator(dynamics=orbit_dynamics, dt=dt) # set noise_std=0 for deterministic trajectory
    key_orbit = jrandom.key(1234)
    r_planet = orbit_dynamics.planet.radius

    # Generate random initial states
    orbit_init_state_specs = [
                {'dist': 'uniform', 'min': r_planet+200, 'max': r_planet+400}, # a (semimajor axis)
                {'dist': 'uniform', 'min': 0.0, 'max': 0.0}, # eccentricity
                {'dist': 'uniform', 'min': 0, 'max': jnp.pi/2}, # inclination
                {'dist': 'uniform', 'min': 0.0, 'max': 2 * jnp.pi}, # right ascension of the ascending node
                {'dist': 'uniform', 'min': 0.0, 'max': 2 * jnp.pi}, # argument of periapsis
                {'dist': 'uniform', 'min': 0.0, 'max': 2 * jnp.pi}, # true anomaly
    ]

    orbit_batch_size = 200
    orbit_states = sample_initial_states(batch_size=orbit_batch_size, key=key_orbit, state_specs=orbit_init_state_specs)
    orbit_init_states = coord.orbital_elements_to_pci(orbit_states,planet=planet)

    # Generate orbits
    start_orbit_batch = time.time()
    orbit_trajs, _ = orbit_system.generate_trajectory_batch(initial_states=orbit_init_states, num_steps=N_orbit, key=key_orbit, batch_size=orbit_batch_size)
    orbit_trajs.block_until_ready()
    print(f"Time: {time.time() - start_orbit_batch:.2f}s")
    return N_orbit, key_orbit, orbit_dynamics, orbit_system, orbit_trajs


@app.cell
def _(orbit_system, orbit_trajs):
    # Visualize a few orbits to verify
    orbit_system.plot_3D(orbit_trajs[80:90])
    return


@app.cell
def _(N_orbit, dt, key_orbit, orbit_dynamics, orbit_trajs):
    # Generate true magnetic field data corresponding to orbits
    chunk_size = 100 # need to chunk otherwise it exhausts memory

    # In PCI coords (true values, no noise)
    b_trajs = orbit_dynamics.mag_model.generate_magnetometer_data(orbit_trajs, dt, N_orbit, key_orbit, 0, jnp.array([0, 0, 0]), chunk_size)

    # Noisy magnetometer values
    noise_std_mag = 1e4 #1e3 #[nT]
    bias = jnp.array([0, 0, 0])
    b_trajs_noisy = orbit_dynamics.mag_model.generate_magnetometer_data(orbit_trajs, dt, N_orbit, key_orbit, noise_std_mag, bias, chunk_size)
    return b_trajs, b_trajs_noisy


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Set dynamics and trajectory generator classes, and conditions for control simulations
    """)
    return


@app.cell
def _(dt):
    # ==================== Spacecraft Dynamics =================== #
    magnetorquer_dynamics = MagnetorquerDynamics()
    system = TrajectoryGenerator(dynamics=magnetorquer_dynamics, dt=dt)
    key = jrandom.key(4567)
    return key, magnetorquer_dynamics, system


@app.cell
def _(
    b_batch,
    estimated_params_batch,
    key,
    magnetorquer_dynamics,
    target_states,
):
    # Set general conditions for single runs
    key1, key2, key3, init_cntrl_key = jrandom.split(key, 4) # split to try out a couple different noises before running batch

    # Define standard deviation of random disturbance torque in simulations
    noise_std_dyn = 1e-6 # [Nm]

    batch_i = 86 # Select orbit/magnetic field/target state from one of the randomly sampled ones from batch

    # Choose b-field (for dynamics propagation)
    b = b_batch[batch_i]

    # Select orbital positions and noisy magnetometer measurements to use w/ learned model
    estimated_params = estimated_params_batch[batch_i]

    state_limits = jnp.array([[-180, 180]]*3 + [[-2,2]]*3)
    control_limits = 0.8*jnp.array([[-1, 1]] * magnetorquer_dynamics.num_controls) # limits for dipole cntrl

    init_state = jnp.array([1, 0, 0, 0, 0.0, 0.0, 0.0])
    target_state = target_states[batch_i]

    quat_start = magnetorquer_dynamics.params["quat_start"]
    quat_goal = target_state[quat_start:quat_start+4]
    return (
        b,
        control_limits,
        estimated_params,
        init_cntrl_key,
        init_state,
        key1,
        noise_std_dyn,
        quat_start,
        state_limits,
        target_state,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Define controller parameters and initialize controllers
    """)
    return


@app.cell
def _(control_limits, magnetorquer_dynamics, state_limits, system):
    # Params for QP MPC
    horizon_qp = 200
    replan_freq_qp = 5
    adapt_freq_qp = 100
    history_qp = 200

    num_steps = 1500 # how long to run sim for

    # QP Solver gains
    Q_qp =  jnp.diag(jnp.array([1e7]*3 + [1e-1]*3))
    Qf_qp = Q_qp
    R_qp = 1e1*jnp.eye(magnetorquer_dynamics.num_controls)

    # Feedback controller gains
    Q_fb = jnp.diag(jnp.array([1e2]*3 + [1e1]*3))
    Qf_fb = Q_fb
    R_fb = 1e1*jnp.eye(magnetorquer_dynamics.num_controls)

    # Initialize controllers
    qp_controller = QPController(system, Q_qp, Qf_qp, R_qp, state_limits, control_limits, horizon_qp)
    fb_controller = FeedbackController(system,Q_fb,Qf_fb,R_fb,control_limits)
    return (
        adapt_freq_qp,
        fb_controller,
        history_qp,
        horizon_qp,
        num_steps,
        qp_controller,
        replan_freq_qp,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Define how learning is going to be used in rollouts
    """)
    return


@app.cell
def _(adapt_freq_qp, history_qp, model_coarse, model_s, planet):
    fixed_full_applicator = LearnedPosToMagMap(model_s, planet)
    fixed_low_applicator = LearnedPosToMagMap(model_coarse, planet)
    adaptation_applicator = AdaptMagModel(model_coarse, planet, last_layer_update, adapt_freq=adapt_freq_qp, history=history_qp, adapt_batch_size=32, adapt_lr=1e-1, adapt_iter=200)
    return adaptation_applicator, fixed_full_applicator, fixed_low_applicator


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Run open-loop solve to check performance of QP solver, check difference between linearized and nonlinear dynamics
    """)
    return


@app.cell
def _(
    b,
    horizon_qp,
    init_cntrl_key,
    init_state,
    key,
    key1,
    magnetorquer_dynamics,
    qp_controller,
    system,
    target_state,
):
    # Open loop QP solve w dipole input
    # Initial traj and cntrl for open loop runs
    init_traj_true = jnp.tile(init_state,(horizon_qp+1,1))
    init_cntrl_true = 0.001*jrandom.normal(init_cntrl_key, shape=((horizon_qp, magnetorquer_dynamics.num_controls)))

    nom_qp_traj, nom_qp_cntrl = qp_controller(init_state, target_state, key, b[:horizon_qp], init_traj_true, init_cntrl_true)
    traj_nl, _ = system.generate_trajectory(initial_state=init_state, target_state=None, key=key1, external_dynamics_params=b[:horizon_qp], control_sequence=nom_qp_cntrl, num_steps=horizon_qp)
    _fig1 = system.plot_traj(jnp.array([nom_qp_traj, traj_nl]), nom_qp_cntrl, target_state)
    _, _fig2 = system.plot_costs(jnp.array([nom_qp_traj, traj_nl]), target_state,legend=["Linear", "True"])
    mo.hstack([_fig1, _fig2])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Generate single closed-loop trajectories
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Generate single closed-loop trajectories for all cases of magnetic field model fidelity that controller has access to:
    1. True magnetic field model (True)
    2. Learned magnetic field model trained on data from full-fidelity magnetic field model (Full)
    3. Learned magnetic field model trained on data from dipole magnetic field model (Dipole)
    4. Dipole-learned magnetic field model with online adaptations (Adapt)
    """)
    return


@app.cell(hide_code=True)
def _():
    single_traj_run_button = mo.ui.run_button(label="Generate single closed-loop trajectories")
    single_traj_run_button
    return (single_traj_run_button,)


@app.cell
def _(
    b,
    fb_controller,
    init_state,
    key1,
    noise_std_dyn,
    num_steps,
    qp_controller,
    replan_freq_qp,
    single_traj_run_button,
    system,
    target_state,
):
    # Closed loop QP solver - True params
    mo.stop(not single_traj_run_button.value, mo.md("Click button to run this cell"))
    traj_true, cntrl_true = system.generate_trajectory(initial_state=init_state, target_state=target_state, key=key1, external_dynamics_params=b, external_dynamics_params_est=None, high_level_controller=qp_controller, low_level_controller=fb_controller, replan_freq=replan_freq_qp, num_steps=num_steps, noise_std=noise_std_dyn)
    system.plot_traj(traj_true, cntrl_true, target_state)
    traj_true_df, _  = system.plot_costs(traj_true,target_state)
    print(jnp.array(traj_true_df["Angle Errors"])[-1])
    return cntrl_true, traj_true


@app.cell
def _(
    b,
    estimated_params,
    fb_controller,
    fixed_full_applicator,
    init_state,
    key,
    noise_std_dyn,
    num_steps,
    qp_controller,
    replan_freq_qp,
    single_traj_run_button,
    system,
    target_state,
):
    # Closed loop QP solver - Learned full-order model
    mo.stop(not single_traj_run_button.value, mo.md("Click button to run this cell"))
    traj_full, cntrl_full = system.generate_trajectory(initial_state=init_state, target_state=target_state, key=key, external_dynamics_params=b, external_dynamics_params_est=estimated_params, high_level_controller=qp_controller, low_level_controller=fb_controller, replan_freq=replan_freq_qp, num_steps=num_steps, noise_std=noise_std_dyn, learning_applicator=fixed_full_applicator)
    return


@app.cell
def _(
    b,
    estimated_params,
    fb_controller,
    fixed_low_applicator,
    init_state,
    key,
    noise_std_dyn,
    num_steps,
    qp_controller,
    replan_freq_qp,
    single_traj_run_button,
    system,
    target_state,
):
    # Closed loop QP solver - Learned dipole model
    mo.stop(not single_traj_run_button.value, mo.md("Click button to run this cell"))
    traj_low, cntrl_low = system.generate_trajectory(initial_state=init_state, target_state=target_state, key=key, external_dynamics_params=b, external_dynamics_params_est=estimated_params, high_level_controller=qp_controller, low_level_controller=fb_controller, replan_freq=replan_freq_qp, num_steps=num_steps, noise_std=noise_std_dyn, learning_applicator=fixed_low_applicator)
    return cntrl_low, traj_low


@app.cell
def _(
    adaptation_applicator,
    b,
    estimated_params,
    fb_controller,
    init_state,
    key,
    noise_std_dyn,
    num_steps,
    qp_controller,
    replan_freq_qp,
    single_traj_run_button,
    system,
    target_state,
):
    # Closed loop QP solver -Learned dipole model w/ adaptation
    mo.stop(not single_traj_run_button.value, mo.md("Click button to run this cell"))
    traj_adapt, cntrl_adapt = system.generate_trajectory(initial_state=init_state, target_state=target_state, key=key, external_dynamics_params=b, external_dynamics_params_est=estimated_params, high_level_controller=qp_controller, low_level_controller=fb_controller, replan_freq=replan_freq_qp, num_steps=num_steps, noise_std=noise_std_dyn, learning_applicator=adaptation_applicator)
    return cntrl_adapt, traj_adapt


@app.cell
def _(
    cntrl_adapt,
    cntrl_low,
    cntrl_true,
    system,
    target_state,
    traj_adapt,
    traj_low,
    traj_true,
):
    # Compare all four cases
    _fig1 = system.plot_traj(jnp.array([traj_true, traj_low, traj_adapt]), jnp.array([cntrl_true, cntrl_low, cntrl_adapt]),target_state=target_state, legend=["True","Fixed", "Adapt"])
    _, _fig2 = system.plot_costs(jnp.array([traj_true, traj_low, traj_adapt]), target_state,legend=["True", "Fixed", "Adapt"])
    mo.hstack([_fig1, _fig2])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Monte Carlo Simulations
     Verify controller by generating batch of closed-loop trajectories with randomized orbits and target states
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Set batch parameters
    """)
    return


@app.cell
def _(b_trajs, b_trajs_noisy, key, orbit_trajs):
    # Generate random initial states and set batch properties
    init_state_specs = [
                {'shape': (4,), 'dist': 'quaternion'}, # quaternion
                {'shape': (3,), 'dist': 'uniform', 'min': 0.0, 'max': 0.0} # angular velocity
    ]

    target_state_specs = [
                {'shape': (4,), 'dist': 'quaternion'}, # quaternion
                {'shape': (3,), 'dist': 'uniform', 'min': 0.0, 'max': 0.0} # angular velocity
    ]

    batch_size = 100

    new_key, init_state_key, targ_state_key = jrandom.split(key,3)
    init_states = sample_initial_states(batch_size=batch_size, key=init_state_key, state_specs=init_state_specs)
    target_states = sample_initial_states(batch_size=batch_size, key=targ_state_key, state_specs=target_state_specs)

    # Get random b-field, always start at beginning of orbit (orbit start points already randomized)
    b_batch_idx = jnp.arange(b_trajs.shape[0])
    new_key, b_key  = jrandom.split(new_key)
    shuffled_indices = jrandom.permutation(b_key, b_batch_idx)
    batch_indices = shuffled_indices[:batch_size]
    b_batch = b_trajs[batch_indices, :, :]
    b_batch_noisy = b_trajs_noisy[batch_indices, :, :]
    orbit_xyz_batch = orbit_trajs[batch_indices, :, :3]
    estimated_params_batch = jnp.concatenate([orbit_xyz_batch, b_batch_noisy],axis=-1)

    num_steps_batch = 1500
    return (
        b_batch,
        batch_size,
        estimated_params_batch,
        num_steps_batch,
        target_states,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Generate batch of trajectories for all (4) cases
    """)
    return


@app.cell(hide_code=True)
def _():
    batch_run_button = mo.ui.run_button(label="Generate batch of trajectories for all cases")
    batch_run_button
    return (batch_run_button,)


@app.cell
def _(
    b_batch,
    batch_run_button,
    batch_size,
    fb_controller,
    init_state,
    key,
    noise_std_dyn,
    num_steps_batch,
    qp_controller,
    replan_freq_qp,
    system,
    target_states,
):
    # QP Solver Batch - true params
    mo.stop(not batch_run_button.value, mo.md("Click button to run this cell"))
    start_batch_qp = time.time()
    trajs_true, cntrls_true= system.generate_trajectory_batch(initial_states=init_state, target_states=target_states, key=key, batch_size=batch_size, external_dynamics_params=b_batch, external_dynamics_params_est=None, high_level_controller=qp_controller, low_level_controller=fb_controller, replan_freq=replan_freq_qp, chunk_size=None, num_steps=num_steps_batch, noise_std=noise_std_dyn)
    trajs_true.block_until_ready()
    print(f"Time: {time.time() - start_batch_qp:.2f}s")
    return (trajs_true,)


@app.cell(disabled=True, hide_code=True)
def _(
    b_batch,
    batch_run_button,
    batch_size,
    estimated_params_batch,
    fb_controller,
    fixed_full_applicator,
    init_state,
    key,
    noise_std_dyn,
    num_steps_batch,
    qp_controller,
    replan_freq_qp,
    system,
    target_states,
):
    # QP Solver Batch - full order learned model
    mo.stop(not batch_run_button.value, mo.md("Click button to run this cell"))
    trajs_full, cntrls_full = system.generate_trajectory_batch(initial_states=init_state, target_states=target_states, key=key, batch_size=batch_size, external_dynamics_params=b_batch, external_dynamics_params_est=estimated_params_batch, high_level_controller=qp_controller, low_level_controller=fb_controller, replan_freq=replan_freq_qp, chunk_size=None, num_steps=num_steps_batch, noise_std=noise_std_dyn, learning_applicator=fixed_full_applicator)
    return


@app.cell
def _(
    b_batch,
    batch_run_button,
    batch_size,
    estimated_params_batch,
    fb_controller,
    fixed_low_applicator,
    init_state,
    key,
    noise_std_dyn,
    num_steps_batch,
    qp_controller,
    replan_freq_qp,
    system,
    target_states,
):
    # QP Solver Batch - low-order model
    mo.stop(not batch_run_button.value, mo.md("Click button to run this cell"))
    trajs_qp_coarse, cntrls_qp_coarse = system.generate_trajectory_batch(initial_states=init_state, target_states=target_states, key=key, batch_size=batch_size, external_dynamics_params=b_batch, external_dynamics_params_est=estimated_params_batch, high_level_controller=qp_controller, low_level_controller=fb_controller, replan_freq=replan_freq_qp, chunk_size=None, num_steps=num_steps_batch, noise_std=noise_std_dyn, learning_applicator=fixed_low_applicator)
    return (trajs_qp_coarse,)


@app.cell
def _(
    adaptation_applicator,
    b_batch,
    batch_run_button,
    batch_size,
    estimated_params_batch,
    fb_controller,
    init_state,
    key,
    noise_std_dyn,
    num_steps_batch,
    qp_controller,
    replan_freq_qp,
    system,
    target_states,
):
    # QP Solver Batch - adaptation
    mo.stop(not batch_run_button.value, mo.md("Click button to run this cell"))
    trajs_qp_adapt, cntrls_qp_adapt = system.generate_trajectory_batch(initial_states=init_state, target_states=target_states, key=key, batch_size=batch_size, external_dynamics_params=b_batch, external_dynamics_params_est=estimated_params_batch, high_level_controller=qp_controller, low_level_controller=fb_controller, replan_freq=replan_freq_qp, num_steps=num_steps_batch, noise_std=noise_std_dyn, learning_applicator=adaptation_applicator)
    return (trajs_qp_adapt,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Save data to ```pandas``` dataframe using ```pickle```
    """)
    return


@app.cell
def _(
    dt,
    horizon_qp,
    planet,
    quat_start,
    target_states,
    trajs_qp_adapt,
    trajs_qp_coarse,
    trajs_true,
):
    # Save data to pandas dataframe
    if planet is Earth:
        if horizon_qp == 50:
            filename = 'data/earth_50.pkl'
        elif horizon_qp == 100:
            filename = 'data/earth_100.pkl'
        elif horizon_qp == 150:
            filename = 'data/earth_150.pkl'
        elif horizon_qp == 200:
            filename = 'earth_200.pkl'
        elif horizon_qp == 250:
            filename = 'earth_250.pkl'

    elif planet is Uranus:
        if horizon_qp == 50:
            filename = 'uranus_50.pkl'
        elif horizon_qp == 100:
            filename = 'uranus_100.pkl'
        elif horizon_qp == 150:
            filename = 'uranus_150.pkl'
        elif horizon_qp == 200:
            filename = 'uranus_200.pkl'
        elif horizon_qp == 250:
            filename = 'uranus_250.pkl'

    df = save_data_to_pd_df(jnp.array([trajs_true, trajs_qp_coarse, trajs_qp_adapt]), target_states=target_states, labels=["True", "Fixed", "Adapt"],dt=dt,quat_start=quat_start,filename=filename)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Plotting
    If you provide a filename to the plotting functions, the plots are saved
    """)
    return


@app.cell
def _():
    # Thresholds for success
    angle_threshold = 15 # phi = 2*arccos(q.T @ q_g)*180/pi
    omega_threshold = 5 # (deg/s)

    # Tolerances for stability
    angle_tol = 10
    omega_tol = 5
    tail_length = 100
    # time_hist_max = 250
    angle_hist_max = 30
    omega_hist_max = 15
    return (
        angle_hist_max,
        angle_threshold,
        angle_tol,
        omega_hist_max,
        omega_threshold,
        omega_tol,
        tail_length,
    )


@app.cell
def _(system, target_states, trajs_qp_adapt):
    _, _fig = system.plot_costs(trajs_qp_adapt, target_state=target_states,plot_stats=True)
    _fig
    return


@app.cell
def _(
    angle_threshold,
    angle_tol,
    omega_threshold,
    omega_tol,
    system,
    tail_length,
    target_states,
    trajs_qp_adapt,
):
    _, _fig = system.plot_violin_and_bar(trajs_qp_adapt, target_states, angle_threshold=angle_threshold, omega_threshold=omega_threshold,angle_stability_tol=angle_tol, omega_stability_tol=omega_tol, tail_length=tail_length, verbose=True)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Or you can plot from the saved dataframe.
    """)
    return


@app.cell(disabled=True, hide_code=True)
def _(angle_threshold, omega_threshold, system):
    # Load dataframes
    earth_df_200 = pd.read_pickle('data/earth_200.pkl')
    uranus_df_200 = pd.read_pickle('data/uranus_200.pkl')

    _ = system.plot_costs(df=earth_df_200, angle_threshold=angle_threshold, omega_threshold=omega_threshold, plot_stats=True)#,filename='earth_mrp')
    return earth_df_200, uranus_df_200


@app.cell(disabled=True, hide_code=True)
def _(
    angle_hist_max,
    angle_threshold,
    angle_tol,
    earth_df_200,
    omega_hist_max,
    omega_threshold,
    omega_tol,
    system,
    tail_length,
    uranus_df_200,
):
    df_list = [earth_df_200, uranus_df_200]
    labels = ["Earth", "Uranus"]
    filename_fig='planets.png'

    system.plot_violin_and_bar(df_list=df_list,angle_threshold=angle_threshold,omega_threshold=omega_threshold,angle_stability_tol=angle_tol, omega_stability_tol=omega_tol, tail_length=tail_length, angle_hist_max=angle_hist_max, omega_hist_max=omega_hist_max, verbose=True, dataset_labels=labels)
    return


if __name__ == "__main__":
    app.run()
