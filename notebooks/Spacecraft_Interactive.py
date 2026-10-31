import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Resolve assets from the notebook, regardless of the launch directory.
    project_root = Path(__file__).resolve().parents[1]
    return mo, np, pd, plt, project_root


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Spacecraft control explorer
    Browse saved results immediately, or prepare a small orbital ensemble and run
    new controller comparisons below. Plot controls only recompute views of completed
    results. Simulation settings are applied together; expensive control rollouts
    require an explicit **Run controller comparison** click. The installed QP solver
    requires CUDA for new controller runs; saved-result exploration works on CPU.

    Angles are in degrees, angular-rate errors in degrees/second, magnetic fields in
    nT, and simulation times in seconds. New runs start with an identity attitude and
    zero angular velocity, with independently sampled target attitudes.
    """)
    return


@app.cell(hide_code=True)
def _(mo, project_root):
    saved_paths = {p.stem: p for p in sorted((project_root / "data").glob("*.pkl"))
                   if p.stem.startswith(("earth_", "uranus_"))}
    saved_dataset = mo.ui.dropdown(list(saved_paths), value="earth_200" if "earth_200" in saved_paths else next(iter(saved_paths), None), label="Saved dataset")
    result_source = mo.ui.radio(["Saved results", "New simulation"], value="Saved results", label="Results source")
    mo.hstack([result_source, saved_dataset])
    return result_source, saved_dataset, saved_paths


@app.cell
def _(pd, saved_dataset, saved_paths):
    saved_frame = pd.read_pickle(saved_paths[saved_dataset.value]) if saved_dataset.value else pd.DataFrame()
    return (saved_frame,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Prepare orbital scenarios
    Apply this form to generate orbit and true/noisy magnetic-field histories.
    The default ensemble is deliberately small. Seeds make scenario generation
    reproducible. Changing planets reloads the matching learned models only when
    needed for a control run.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    environment_form = mo.ui.batch(mo.md("""
    Planet: {planet} · Seed: {seed} · Scenarios: {count}

    Time step (s): {dt} · Orbit steps: {orbit_steps}

    Altitude range (km): {altitude} · Maximum inclination (deg): {inclination}

    Magnetometer noise standard deviation (nT): {mag_noise}
    """), {
        "planet": mo.ui.dropdown(["Earth", "Uranus"], value="Earth"),
        "seed": mo.ui.number(0, 1000000, value=1234, step=1),
        "count": mo.ui.number(1, 32, value=8, step=1),
        "dt": mo.ui.number(0.01, 1.0, value=0.1, step=0.01),
        "orbit_steps": mo.ui.number(500, 10000, value=2000, step=100),
        "altitude": mo.ui.range_slider(100, 2000, value=[200, 400], step=10, show_value=True),
        "inclination": mo.ui.slider(0, 180, value=90, step=5, show_value=True),
        "mag_noise": mo.ui.number(0, 100000, value=10000, step=100),
    }).form(submit_button_label="Prepare scenarios")
    environment_form
    return (environment_form,)


@app.cell
def _(np, pd):
    def build_environment(settings):
        import os
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        import jax
        import jax.numpy as jnp
        import jax.random as jr
        from jaxitude.utils import TrajectoryGenerator, sample_initial_states, coord
        from jaxitude.dynamics import OrbitDynamics, Earth, Uranus

        jax.config.update("jax_default_matmul_precision", "highest")
        jax.config.update("jax_enable_x64", True)
        planet = {"Earth": Earth, "Uranus": Uranus}[settings["planet"]]
        dynamics = OrbitDynamics(Planet=planet)
        system = TrajectoryGenerator(dynamics=dynamics, dt=settings["dt"])
        orbit_key, field_key, target_key, run_key = jr.split(jr.key(int(settings["seed"])), 4)
        radius = dynamics.planet.radius
        low, high = settings["altitude"]
        specs = [
            {"dist": "uniform", "min": radius + low, "max": radius + high},
            {"dist": "uniform", "min": 0.0, "max": 0.0},
            {"dist": "uniform", "min": 0.0, "max": np.deg2rad(settings["inclination"])},
            *[{"dist": "uniform", "min": 0.0, "max": 2 * jnp.pi} for _ in range(3)],
        ]
        count, steps = int(settings["count"]), int(settings["orbit_steps"])
        states = sample_initial_states(batch_size=count, key=orbit_key, state_specs=specs)
        initial = coord.orbital_elements_to_pci(states, planet=planet)
        orbits, _ = system.generate_trajectory_batch(initial_states=initial, num_steps=steps, key=orbit_key, batch_size=count)
        fields = dynamics.mag_model.generate_magnetometer_data(orbits, settings["dt"], steps, field_key, 0, jnp.zeros(3), min(count, 8))
        # Independent noise key; estimated parameters actually contain noisy measurements.
        noisy_fields = fields + settings["mag_noise"] * jr.normal(field_key, fields.shape)
        targets = sample_initial_states(batch_size=count, key=target_key, state_specs=[
            {"shape": (4,), "dist": "quaternion"},
            {"shape": (3,), "dist": "uniform", "min": 0.0, "max": 0.0},
        ])
        fields.block_until_ready()
        return dict(settings=dict(settings), planet=planet, orbits=orbits, fields=fields,
                    estimates=jnp.concatenate([orbits[..., :3], noisy_fields], axis=-1),
                    targets=targets, key=run_key)


    def run_comparison(environment, settings, scenario, root):
        import time
        import jax.numpy as jnp
        import jax.random as jr
        from jaxitude.utils import (TrajectoryGenerator, load_model, LearnedPosToMagMap,
                                    AdaptMagModel, last_layer_update)
        from jaxitude.dynamics import MagnetorquerDynamics
        from jaxitude.controllers import QPController, FeedbackController

        dynamics = MagnetorquerDynamics()
        system = TrajectoryGenerator(dynamics=dynamics, dt=environment["settings"]["dt"])
        limits = settings["dipole_limit"] * jnp.array([[-1, 1]] * dynamics.num_controls)
        q = jnp.diag(jnp.array([10.0 ** settings["q_angle"]] * 3 + [10.0 ** settings["q_rate"]] * 3))
        r = 10.0 ** settings["r_control"] * jnp.eye(dynamics.num_controls)
        qp = QPController(system, q, q, r, jnp.array([[-180, 180]] * 3 + [[-2, 2]] * 3), limits, int(settings["horizon"]))
        q_fb = jnp.diag(jnp.array([1e2] * 3 + [1e1] * 3))
        fb = FeedbackController(system, q_fb, q_fb, 1e1 * jnp.eye(dynamics.num_controls), limits)
        indices = [scenario] if settings["mode"] == "Single scenario" else list(range(len(environment["targets"])))
        models = {}
        prefix = environment["settings"]["planet"].lower()
        for name, suffix in [("Full", ""), ("Fixed", "_coarse")]:
            if name in settings["models"] or (name == "Fixed" and "Adapt" in settings["models"]):
                models[name], _ = load_model(filename=str(root / "src" / "jaxitude" / "models" / f"{prefix}_b_4d{suffix}.eqx"))
        results, timings = {}, []
        # Each case receives the same random keys for a fair disturbance comparison.
        keys = jr.split(environment["key"], len(environment["targets"]))
        for name in settings["models"]:
            applicator = None
            if name in ("Full", "Fixed"):
                applicator = LearnedPosToMagMap(models[name], environment["planet"])
            elif name == "Adapt":
                applicator = AdaptMagModel(models["Fixed"], environment["planet"], last_layer_update,
                    adapt_freq=int(settings["adapt_freq"]), history=int(settings["history"]),
                    adapt_batch_size=min(32, int(settings["history"])), adapt_lr=10.0 ** settings["learning_rate"],
                    adapt_iter=int(settings["adapt_iterations"]))
            start = time.perf_counter()
            for index in indices:
                trajectory, controls = system.generate_trajectory(
                    initial_state=jnp.array([1., 0., 0., 0., 0., 0., 0.]),
                    target_state=environment["targets"][index], key=keys[index],
                    external_dynamics_params=environment["fields"][index],
                    external_dynamics_params_est=None if name == "True" else environment["estimates"][index],
                    high_level_controller=qp, low_level_controller=fb, learning_applicator=applicator,
                    replan_freq=int(settings["replan"]), num_steps=int(settings["steps"]),
                    noise_std=10.0 ** settings["torque_noise"])
                trajectory.block_until_ready()
                results[name, index] = (np.asarray(trajectory), np.asarray(controls))
            timings.append({"Group": name, "Runtime (s, includes compilation)": time.perf_counter() - start})
        rows = []
        for (name, index), (trajectory, controls) in results.items():
            target = np.asarray(environment["targets"][index])
            angle = np.degrees(2 * np.arccos(np.clip(np.abs(trajectory[:, :4] @ target[:4]), 0, 1)))
            omega = np.degrees(np.linalg.norm(trajectory[:, 4:] - target[4:], axis=-1))
            rows.append(pd.DataFrame({"Group": name, "Trial": index,
                "Time": np.arange(len(trajectory)) * environment["settings"]["dt"],
                "Angle Errors": angle, "Omega Errors": omega}))
        return {"frame": pd.concat(rows, ignore_index=True), "raw": results,
                "timings": pd.DataFrame(timings), "settings": dict(settings),
                "environment_settings": environment["settings"]}

    return build_environment, run_comparison


@app.cell
def _(build_environment, environment_form, mo):
    mo.stop(environment_form.value is None, mo.md("Submit **Prepare scenarios** to enable new simulations. Saved results remain available above and below."))
    with mo.status.spinner(title="Generating orbital scenarios and magnetic fields"):
        environment = build_environment(environment_form.value)
    return (environment,)


@app.cell(hide_code=True)
def _(environment, mo):
    scenario_picker = mo.ui.slider(0, len(environment["targets"]) - 1, value=0, step=1, show_value=True, label="Scenario index")
    scenario_picker
    return (scenario_picker,)


@app.cell
def _(environment, mo, np, plt, scenario_picker):
    _index = int(scenario_picker.value)
    _orbit = np.asarray(environment["orbits"][_index])
    _field = np.asarray(environment["fields"][_index])
    _time = np.arange(len(_field)) * environment["settings"]["dt"]
    _fig = plt.figure(figsize=(11, 4))
    _ax = _fig.add_subplot(121, projection="3d")
    _ax.plot(*_orbit[:, :3].T)
    _ax.set(xlabel="PCI x (km)", ylabel="PCI y (km)", zlabel="PCI z (km)", title=f"Orbit {_index}")
    _ax2 = _fig.add_subplot(122)
    _ax2.plot(_time, _field, label=["Bx", "By", "Bz"])
    _ax2.set(xlabel="Time (s)", ylabel="Magnetic field (nT)")
    _ax2.legend()
    plt.close(_fig)
    mo.vstack([_fig, mo.md(f"Target quaternion: `{np.asarray(environment['targets'][_index, :4]).round(4).tolist()}`")])
    return


@app.cell(hide_code=True)
def _(mo):
    controller_form = mo.ui.batch(mo.md("""
    ## Controller comparison settings
    Run mode: {mode} · Cases: {models}

    Simulation steps: {steps} · MPC horizon: {horizon} · Replan every N steps: {replan}

    Q attitude exponent: {q_angle} · Q rate exponent: {q_rate} · R exponent: {r_control}

    Dipole limit (A m²): {dipole_limit} · Disturbance torque log10(N m): {torque_noise}

    **Online adaptation** — frequency (steps): {adapt_freq} · History (steps): {history}

    Learning-rate exponent: {learning_rate} · Update iterations: {adapt_iterations}

    Weights and noise use powers of ten, e.g. exponent −6 means 0.000001.
    Ensemble mode runs every prepared scenario, sequentially to limit memory usage.
    """), {
        "mode": mo.ui.dropdown(["Single scenario", "Ensemble"], value="Single scenario"),
        "models": mo.ui.multiselect(["True", "Full", "Fixed", "Adapt"], value=["True", "Fixed", "Adapt"]),
        "steps": mo.ui.number(100, 5000, value=500, step=100),
        "horizon": mo.ui.dropdown([50, 100, 150, 200, 250], value=200),
        "replan": mo.ui.number(1, 100, value=5, step=1),
        "q_angle": mo.ui.slider(2, 10, value=7, step=1, show_value=True),
        "q_rate": mo.ui.slider(-4, 4, value=-1, step=1, show_value=True),
        "r_control": mo.ui.slider(-3, 5, value=1, step=1, show_value=True),
        "dipole_limit": mo.ui.number(0.01, 10, value=0.8, step=0.01),
        "torque_noise": mo.ui.slider(-10, -3, value=-6, step=1, show_value=True),
        "adapt_freq": mo.ui.number(1, 500, value=100, step=1),
        "history": mo.ui.number(32, 1000, value=200, step=1),
        "learning_rate": mo.ui.slider(-5, 0, value=-1, step=1, show_value=True),
        "adapt_iterations": mo.ui.number(1, 500, value=200, step=1),
    }).form(submit_button_label="Apply controller settings")
    controller_form
    return (controller_form,)


@app.cell(hide_code=True)
def _(mo):
    run_button = mo.ui.run_button(label="Run controller comparison")
    run_button
    return (run_button,)


@app.cell
def _(mo):
    completed_result, set_completed_result = mo.state(None)
    return completed_result, set_completed_result


@app.cell
def _(
    controller_form,
    environment,
    mo,
    project_root,
    run_button,
    run_comparison,
    scenario_picker,
    set_completed_result,
):
    mo.stop(not run_button.value, mo.md("Apply controller settings, then click **Run controller comparison**."))
    mo.stop(controller_form.value is None, mo.md("Apply controller settings first."))
    _settings = controller_form.value
    mo.stop(not _settings["models"], mo.md("Select at least one model case."))
    mo.stop(_settings["steps"] + _settings["horizon"] > environment["settings"]["orbit_steps"],
            mo.md("Prepare more orbit steps: the field history must cover simulation steps plus the MPC horizon."))
    mo.stop(_settings["replan"] > _settings["horizon"], mo.md("Replanning interval must not exceed the MPC horizon."))
    with mo.status.spinner(title="Running controller comparison (first run includes JAX compilation)"):
        simulation_result = run_comparison(environment, _settings, int(scenario_picker.value), project_root)
    set_completed_result(simulation_result)
    mo.vstack([mo.ui.table(simulation_result["timings"]), mo.md(f"Completed settings: `{simulation_result['settings']}`")])
    return


@app.cell
def _(completed_result, mo, result_source, saved_frame):
    # State makes completed results available independently of stopped run cells.
    if result_source.value == "Saved results":
        active_frame = saved_frame.copy()
    else:
        _completed = completed_result()
        mo.stop(_completed is None, mo.md("Run a controller comparison first, then select New simulation."))
        active_frame = _completed["frame"].copy()
    mo.stop(active_frame.empty, mo.md("No results available for this source."))
    if "Trial" not in active_frame:
        # Legacy files store each trial contiguously, restarting Time at zero.
        active_frame["Trial"] = active_frame.groupby("Group", sort=False)["Time"].transform(lambda s: s.diff().fillna(-1).le(0).cumsum() - 1).astype(int)
    return (active_frame,)


@app.cell(hide_code=True)
def _(active_frame, mo):
    plot_models = mo.ui.multiselect(list(active_frame["Group"].unique()), value=list(active_frame["Group"].unique()), label="Visible model cases")
    plot_trial = mo.ui.dropdown(sorted(active_frame["Trial"].unique().tolist()), value=int(active_frame["Trial"].min()), label="Trial")
    _tmax = float(active_frame["Time"].max())
    time_window = mo.ui.range_slider(0, _tmax, value=[0, _tmax], step=max(_tmax / 1000, 0.001), show_value=True, label="Time window (s)")
    mo.vstack([mo.md("## Explore completed results"), mo.hstack([plot_models, plot_trial]), time_window])
    return plot_models, plot_trial, time_window


@app.cell(hide_code=True)
def _(mo):
    angle_threshold = mo.ui.slider(0.1, 90, value=15, step=0.1, show_value=True, label="Success: angle (deg)")
    omega_threshold = mo.ui.slider(0.1, 30, value=5, step=0.1, show_value=True, label="Success: rate (deg/s)")
    angle_tolerance = mo.ui.slider(0.1, 30, value=10, step=0.1, show_value=True, label="Stability: angle tolerance (deg)")
    omega_tolerance = mo.ui.slider(0.1, 20, value=5, step=0.1, show_value=True, label="Stability: rate tolerance (deg/s)")
    tail_samples = mo.ui.slider(5, 500, value=100, step=5, show_value=True, label="Tail / minimum stable samples")
    mo.vstack([mo.hstack([angle_threshold, omega_threshold]), mo.hstack([angle_tolerance, omega_tolerance]), tail_samples,
        mo.md("Success requires tail-mean angle and rate errors below their thresholds. Stability requires the final contiguous segment to stay within both tolerances of the tail means for at least the requested sample count. Summary statistics always use full trajectories, independent of the plot time window.")])
    return (
        angle_threshold,
        angle_tolerance,
        omega_threshold,
        omega_tolerance,
        tail_samples,
    )


@app.cell
def _(
    active_frame,
    angle_threshold,
    mo,
    omega_threshold,
    plot_models,
    plot_trial,
    plt,
    time_window,
):
    mo.stop(not plot_models.value, mo.md("Select at least one case to display."))
    _view = active_frame.loc[active_frame["Group"].isin(plot_models.value) & (active_frame["Trial"] == plot_trial.value) & active_frame["Time"].between(*time_window.value)]
    _fig, _axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for _name, _group in _view.groupby("Group", sort=False):
        _axes[0].plot(_group["Time"], _group["Angle Errors"], label=_name)
        _axes[1].plot(_group["Time"], _group["Omega Errors"], label=_name)
    for _ax, _threshold, _label in zip(_axes, [angle_threshold.value, omega_threshold.value], ["Angle error (deg)", "Rate error (deg/s)"]):
        _ax.axhline(_threshold, linestyle="--", color="gray", label="Success threshold")
        _ax.set_ylabel(_label)
        _ax.legend()
    _axes[1].set_xlabel("Time (s)")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(
    active_frame,
    angle_threshold,
    angle_tolerance,
    np,
    omega_threshold,
    omega_tolerance,
    pd,
    tail_samples,
):
    _rows = []
    for (_name, _trial), _group in active_frame.groupby(["Group", "Trial"], sort=False):
        _g = _group.sort_values("Time")
        _a, _w = _g["Angle Errors"].to_numpy(), _g["Omega Errors"].to_numpy()
        _tail = int(tail_samples.value)
        _am, _wm = np.mean(_a[-_tail:]), np.mean(_w[-_tail:])
        _inside = (np.abs(_a - _am) < angle_tolerance.value) & (np.abs(_w - _wm) < omega_tolerance.value)
        _stable = np.logical_and.accumulate(_inside[::-1])[::-1]
        _settled = bool(_stable.sum() >= _tail)
        _rows.append({"Group": _name, "Trial": _trial, "Tail angle (deg)": _am, "Tail rate (deg/s)": _wm,
            "Success": bool(_am < angle_threshold.value and _wm < omega_threshold.value),
            "Stable": _settled, "Settling time (s)": float(_g["Time"].iloc[np.argmax(_stable)]) if _settled else np.nan})
    trial_statistics = pd.DataFrame(_rows)
    return (trial_statistics,)


@app.cell(hide_code=True)
def _(mo, plot_models, plt, trial_statistics):
    mo.stop(not plot_models.value, mo.md("Select cases to view summary statistics."))
    _selected = trial_statistics.loc[trial_statistics["Group"].isin(plot_models.value)]
    _summary = _selected.groupby("Group", sort=False).agg(Trials=("Trial", "count"), **{
        "Success (%)": ("Success", lambda s: 100 * s.mean()),
        "Stable (%)": ("Stable", lambda s: 100 * s.mean()),
        "Median settling time (s)": ("Settling time (s)", "median"),
        "Mean tail angle (deg)": ("Tail angle (deg)", "mean")}).reset_index()
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    _axes[0].bar(_summary["Group"], _summary["Success (%)"])
    _axes[0].set(ylabel="Success (%)", ylim=(0, 100))
    _names, _values = [], []
    for _name, _g in _selected.groupby("Group", sort=False):
        _names.append(_name)
        _values.append(_g["Tail angle (deg)"].dropna().to_numpy())
    # A single scenario has no distribution; show points instead of a violin.
    for _i, (_name, _v) in enumerate(zip(_names, _values), 1):
        if len(_v) > 1 and _v.max() > _v.min():
            _axes[1].violinplot([_v], positions=[_i], showmedians=True)
        else:
            _axes[1].scatter([_i] * len(_v), _v)
    _axes[1].set_xticks(range(1, len(_names) + 1), _names)
    _axes[1].set_ylabel("Tail angle error (deg)")
    _fig.tight_layout()
    plt.close(_fig)
    mo.vstack([mo.ui.table(_summary), _fig,
        mo.download(data=_selected.to_csv(index=False).encode(), filename="spacecraft_trial_statistics.csv", label="Download trial statistics (CSV)")])
    return


@app.cell(hide_code=True)
def _(active_frame, mo):
    mo.download(data=active_frame.to_csv(index=False).encode(), filename="spacecraft_results.csv", label="Download current results (CSV)")
    return


@app.cell(hide_code=True)
def _(
    completed_result,
    mo,
    np,
    plot_models,
    plot_trial,
    plt,
    result_source,
    time_window,
):
    mo.stop(result_source.value != "New simulation")
    _result = completed_result()
    mo.stop(_result is None)
    _fig, _ax = plt.subplots(figsize=(10, 3))
    for _name in plot_models.value:
        _pair = _result["raw"].get((_name, plot_trial.value))
        if _pair is not None:
            _controls = _pair[1]
            _time = np.arange(len(_controls)) * _result["environment_settings"]["dt"]
            _mask = (_time >= time_window.value[0]) & (_time <= time_window.value[1])
            _ax.plot(_time[_mask], np.linalg.norm(_controls[_mask], axis=-1), label=_name)
    _ax.set(xlabel="Time (s)", ylabel="Dipole norm (A m²)", title="Control effort for selected trial")
    if _ax.lines:
        _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


if __name__ == "__main__":
    app.run()
