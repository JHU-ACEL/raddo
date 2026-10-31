# /// script
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "pandas",
#     "seaborn",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(app_title="Adaptive Magnetorquer-Only Control")


@app.cell(hide_code=True)
def _():
    import types
    import urllib.request

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    def _load_module(url: str, name: str = "remote_funcs") -> types.ModuleType:
        source = urllib.request.urlopen(url).read().decode("utf-8")
        mod = types.ModuleType(name)
        exec(compile(source, url, "exec"), mod.__dict__)
        return mod

    plot_fn_url = "https://raw.githubusercontent.com/pschwartz26/test_marimo/refs/heads/main/plotting_numpy.py"
    plot_utils = _load_module(plot_fn_url)

    sns.set_theme(context="paper", style="whitegrid", font="serif", font_scale=1.7)
    plt.rcParams["figure.max_open_warning"] = 0  # figures are closed explicitly after rendering
    return mo, np, pd, plot_utils, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Interactive analysis of slew-maneuver results for the **Earth** and **Uranus** datasets
    across several horizons, plus a **torque vs. dipole** comparison. Every section has its own
    controls, and plots and summaries update as you change them.

    | Section | What it shows |
    |:--|:--|
    | **1 · Final-state statistics** | Violin plots and success / stability statistics, Earth vs. Uranus |
    | **2 · Error trajectories** | Angle and angular-velocity error histories, Earth vs. Uranus |
    | **3 · Torque vs. dipole** | Angle and angular-velocity error histories for the torque comparison |
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, pd):
    import io
    import warnings
    from urllib.request import urlopen as _urlopen

    # Compact, pre-processed copies of the data (created once, offline, with
    # build_compact_data.py). They are far smaller and faster to load than the original
    # pickles. After uploading the .npz files somewhere that serves them with CORS
    # (e.g. GitHub -> raw.githubusercontent.com), put the folder URL here.
    # Leave as None to fall back to the original pickles.
    COMPACT_BASE = "https://github.com/JHU-ACEL/uranus-mpc-data/blob/main/compact" 

    PICKLE_BASE = "https://raw.githubusercontent.com/JHU-ACEL/uranus-mpc-data/main/data"
    HORIZONS = [50, 100, 150, 200, 250]

    def _frame_from_compact(z):
        """Rebuild the long-format DataFrame the plotting functions expect."""
        groups = [str(g) for g in z["groups"]]
        angle, omega = z["angle"], z["omega"]
        n_groups, batch, n_steps = angle.shape
        n_rows = n_groups * batch * n_steps
        return pd.DataFrame({
            "Group": np.repeat(np.array(groups), batch * n_steps),
            "Time": np.tile(np.asarray(z["time"], dtype=float), n_groups * batch),
            "Quaternion Errors": np.full(n_rows, np.nan),  # not stored, never plotted
            "Angle Errors": angle.reshape(-1).astype(float),
            "Omega Errors": omega.reshape(-1).astype(float),
        })

    def _load(name, pickle_name, drop_full):
        df = None
        if COMPACT_BASE:
            try:
                with _urlopen(f"{COMPACT_BASE.rstrip('/')}/{name}.npz") as resp:
                    df = _frame_from_compact(np.load(io.BytesIO(resp.read()), allow_pickle=False))
            except Exception as err:  # fall back rather than break the page
                warnings.warn(f"Compact data for {name!r} unavailable ({err}); using the pickle instead.")
        if df is None:
            df = pd.read_pickle(f"{PICKLE_BASE}/{pickle_name}")
        return df.loc[df["Group"] != "Full"] if drop_full else df

    _cache = {}

    def get_df(body, horizon):
        """Earth or Uranus trajectories for one horizon ("Full" group removed). Cached."""
        key = f"{body}_{horizon}"
        if key not in _cache:
            with mo.status.spinner(title=f"Loading {body.title()} · N = {horizon}…"):
                _cache[key] = _load(key, f"{key}.pkl", drop_full=True)
        return _cache[key]

    def get_torque():
        """Torque-vs-dipole trajectories (all groups, as stored). Cached."""
        if "torque_df" not in _cache:
            with mo.status.spinner(title="Loading torque data…"):
                _cache["torque_df"] = _load("torque_df", "torque_v_dipole.pkl", drop_full=False)
        return _cache["torque_df"]

    return HORIZONS, get_df, get_torque


@app.cell(hide_code=True)
def _(mo, np, plot_utils, plt):
    _THRESHOLD_LABEL = r"$\text{threshold}$"  # label plot_costs gives its own threshold line
    _METRICS = ("Principal angle error", "Angular velocity error")

    def select_groups(df, groups):
        """Keep only the requested groups."""
        return df.loc[df["Group"].isin(list(groups))]

    def infer_shape(df):
        """Return (batch_size, n_steps) for a long-format trajectory DataFrame."""
        n_steps = int(df["Time"].nunique())
        rows_per_group = int(df.groupby("Group", sort=False).size().iloc[0])
        return rows_per_group // n_steps, n_steps

    def describe_view(compare_groups, plot_stats):
        if compare_groups and plot_stats:
            return "Mean trajectory of each group, overlaid on shared axes."
        if compare_groups:
            return "Every trajectory, coloured by group, on shared axes."
        if plot_stats:
            return "One slide per group (use the arrows): max, 95th / 50th percentile bands and mean."
        return "One slide per group (use the arrows): every individual trajectory."

    def _decorate(ax, threshold, t_lo, t_hi, show_threshold, fit_y):
        """Apply time window, threshold line and (optionally) y-limits fitted to the window."""
        # plot_costs only draws a threshold in one mode, so remove it and manage it here.
        for line in list(ax.get_lines()):
            if line.get_label() == _THRESHOLD_LABEL:
                line.remove()

        ax.set_xlim(t_lo, t_hi)
        if show_threshold:
            ax.axhline(threshold, color="black", linestyle="--", linewidth=1.5, label="Threshold")

        if fit_y:
            y_top = 0.0
            for line in ax.get_lines():
                if line.get_label() == "Threshold":
                    continue
                x = np.asarray(line.get_xdata(), dtype=float)
                y = np.asarray(line.get_ydata(), dtype=float)
                y = y[(x >= t_lo) & (x <= t_hi)]
                y = y[np.isfinite(y)]
                if y.size:
                    y_top = max(y_top, float(y.max()))
            if show_threshold:
                y_top = max(y_top, float(threshold))
            if y_top > 0:
                ax.set_ylim(0, y_top * 1.08)
        elif show_threshold:
            ax.set_ylim(top=max(ax.get_ylim()[1], float(threshold) * 1.08))

        # Rebuild the legend so it reflects the current set of lines.
        old = ax.get_legend()
        if old is not None:
            old.remove()
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc="upper right",
                      frameon=True, framealpha=0.95, edgecolor="black")

    def render_costs(df, *, batch_size, n_steps, batch_idx, angle_thr, omega_thr,
                     t_window, plot_stats, compare_groups, show_thresholds, fit_y):
        """Run plot_costs and return {row_key: (angle_fig, omega_fig)}."""
        t_lo, t_hi = t_window
        _, figs = plot_utils.plot_costs(
            df=df, batch_size=batch_size, batch_idx=batch_idx, N=n_steps,
            title=None, legend=None,
            angle_threshold=angle_thr, omega_threshold=omega_thr,
            time_max=t_hi, plot_stats=plot_stats, compare_groups=compare_groups,
            filename=None,
        )
        thresholds = (angle_thr, omega_thr)
        for k, fig in enumerate(figs):
            _decorate(fig.axes[0], thresholds[k % 2], t_lo, t_hi, show_thresholds, fit_y)

        keys = ["All groups"] if compare_groups else list(df["Group"].unique())
        return {key: (figs[2 * i], figs[2 * i + 1]) for i, key in enumerate(keys)}

    def control_grid(items, cols=2):
        """Lay controls out in rows of `cols` equal-width columns (holds up at any page width)."""
        rows = []
        for i in range(0, len(items), cols):
            chunk = list(items[i:i + cols])
            chunk += [mo.md("")] * (cols - len(chunk))
            rows.append(mo.hstack(chunk, widths="equal", align="start", gap=2))
        return mo.vstack(rows, gap=0.75)

    def _caption(text, fig):
        return mo.vstack([mo.md(f"**{text}**"), fig], align="center", gap=0.25)

    def cost_layout(results, label_sources=True):
        """Arrange cost figures so the grid matches exactly what is being displayed.

        results: {source: {row_key: (angle_fig, omega_fig)}}
        - several sources (e.g. Earth & Uranus): one row per metric, one column per source
        - one source: angle and omega side by side
        - groups shown separately: one carousel slide per group
        """
        sources = list(results)
        row_keys = list(results[sources[0]])
        blocks = []
        for key in row_keys:
            if len(sources) > 1:
                rows = [
                    mo.hstack(
                        [_caption(f"{s} · {_METRICS[m]}", results[s][key][m]) for s in sources],
                        justify="center", gap=1, wrap=True,
                    )
                    for m in (0, 1)
                ]
            else:
                s = sources[0]
                prefix = f"{s} · " if label_sources else ""
                rows = [
                    mo.hstack(
                        [_caption(prefix + _METRICS[m], results[s][key][m]) for m in (0, 1)],
                        justify="center", gap=1, wrap=True,
                    )
                ]
            if key != "All groups":
                rows = [mo.md(f"#### {key}"), *rows]
            blocks.append(mo.vstack(rows, gap=1))
        # Separate groups are paged through in a carousel; a single block needs no wrapper.
        return mo.carousel(blocks) if len(blocks) > 1 else blocks[0]

    def cost_view(frames, *, compare_groups, batch_idx, angle_thr, omega_thr, t_window,
                  plot_stats, show_thresholds, fit_y, label_sources=True):
        """Build the cost-plot view for {source: DataFrame}, deferring figure creation.

        Figures are the expensive part (especially in the browser), so when groups are shown
        separately each carousel slide is wrapped in mo.lazy and only drawn when it is first
        visible, instead of drawing every group up front.
        """

        def _build(group_names):
            per_source = {}
            for source, df in frames.items():
                sub = df if group_names is None else select_groups(df, group_names)
                batch, n_steps = infer_shape(sub)
                per_source[source] = render_costs(
                    sub, batch_size=batch, n_steps=n_steps,
                    batch_idx=None if batch_idx is None else min(batch_idx, batch - 1),
                    angle_thr=angle_thr, omega_thr=omega_thr, t_window=t_window,
                    plot_stats=plot_stats, compare_groups=compare_groups,
                    show_thresholds=show_thresholds, fit_y=fit_y,
                )
            layout = cost_layout(per_source, label_sources)
            for pairs in per_source.values():
                for pair in pairs.values():
                    for fig in pair:
                        plt.close(fig)
            return layout

        if compare_groups:
            return _build(None)
        groups = list(dict.fromkeys(g for df in frames.values() for g in df["Group"].unique()))
        if len(groups) == 1:
            return _build(groups)
        return mo.carousel([
            mo.lazy(lambda g=g: _build([g]), show_loading_indicator=True) for g in groups
        ])

    return control_grid, cost_view, describe_view, infer_shape, select_groups


@app.cell(hide_code=True)
def _(mo):
    get_violin_reset, _set_violin_reset = mo.state(0)
    get_cost_reset, _set_cost_reset = mo.state(0)
    get_torque_reset, _set_torque_reset = mo.state(0)

    violin_reset_btn = mo.ui.button(
        label="↺ Reset section controls",
        on_click=lambda _: _set_violin_reset(lambda n: n + 1),
    )
    cost_reset_btn = mo.ui.button(
        label="↺ Reset section controls",
        on_click=lambda _: _set_cost_reset(lambda n: n + 1),
    )
    torque_reset_btn = mo.ui.button(
        label="↺ Reset section controls",
        on_click=lambda _: _set_torque_reset(lambda n: n + 1),
    )
    return (
        cost_reset_btn,
        get_cost_reset,
        get_torque_reset,
        get_violin_reset,
        torque_reset_btn,
        violin_reset_btn,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 1 · Final-State Statistics
    "
        Distributions of slew time and converged error across trajectories, Earth vs. Uranus.
        Use **Interact with data** beneath the plots to change the horizon, groups, criteria and panels.
    """)
    return


@app.cell(hide_code=True)
def _(HORIZONS, get_df, get_violin_reset, mo):
    get_violin_reset()  # re-running this cell after a reset restores every default

    _default_h = 200 if 200 in HORIZONS else HORIZONS[-1]
    _groups = [g for g in get_df("earth", _default_h)["Group"].unique() if g != "Full"]
    _panels = ["Slew time", "Final angle error", "Final angular velocity", "Success rate"]

    # Default comparison: "Adapt" against "Fixed" when those groups exist
    _by_lower = {str(g).lower(): g for g in _groups}
    _base_default = _by_lower.get("fixed", _groups[0])
    _cmp_default = _by_lower.get("adapt", next((g for g in _groups if g != _base_default), _groups[0]))

    _slider = dict(show_value=True, debounce=True, full_width=True)

    v_horizon = mo.ui.dropdown(
        options={f"N = {h}": h for h in HORIZONS}, value=f"N = {_default_h}",
        label="Horizon", full_width=True,
    )
    v_groups = mo.ui.multiselect(options=_groups, value=_groups, label="Groups", full_width=True)
    v_panels = mo.ui.multiselect(options=_panels, value=_panels[:3], label="Panels", full_width=True)

    v_compare = mo.ui.dropdown(options=_groups, value=_cmp_default, label="Compare group", full_width=True)
    v_baseline = mo.ui.dropdown(options=_groups, value=_base_default, label="Against baseline", full_width=True)

    v_angle_thr = mo.ui.slider(0.1, 90, value=15.0, step=0.1, label="Success: angle (deg)", **_slider)
    v_omega_thr = mo.ui.slider(0.1, 30, value=5.0, step=0.1, label="Success: rate (deg/s)", **_slider)

    v_angle_tol = mo.ui.slider(0.1, 30, value=10.0, step=0.1, label="Stability: angle tolerance (deg)", **_slider)
    v_omega_tol = mo.ui.slider(0.1, 20, value=5.0, step=0.1, label="Stability: rate tolerance (deg/s)", **_slider)

    v_angle_cap = mo.ui.slider(1, 180, value=15, step=1, label="Angle axis limit (deg)", **_slider)
    v_omega_cap = mo.ui.slider(0.5, 30, value=5.0, step=0.5, label="Rate axis limit (deg/s)", **_slider)
    return (
        v_angle_cap,
        v_angle_thr,
        v_angle_tol,
        v_baseline,
        v_compare,
        v_groups,
        v_horizon,
        v_omega_cap,
        v_omega_thr,
        v_omega_tol,
        v_panels,
    )


@app.cell(hide_code=True)
def _(get_df, mo, np, v_horizon):
    # Controls whose range depends on the selected horizon
    _h = v_horizon.value
    _frames = [get_df("earth", _h), get_df("uranus", _h)]
    _t_max = max(float(f["Time"].max()) for f in _frames)
    _n_steps = int(_frames[0]["Time"].nunique())

    _steps = list(dict.fromkeys(float(t) for t in np.round(np.linspace(0, _t_max, 101)[1:], 2)))
    v_time_cap = mo.ui.slider(
        steps=_steps, value=_steps[-1], show_value=True, debounce=True, full_width=True,
        label="Slew-time axis limit (s)",
    )

    # The stability window cannot exceed the number of samples in a trajectory
    _tail_stop = max(5, (_n_steps // 5) * 5)
    _tail_default = min(100, max(5, (_n_steps // 2 // 5) * 5))
    v_tail = mo.ui.slider(
        5, _tail_stop, value=_tail_default, step=5, show_value=True, debounce=True,
        full_width=True, label="Stability: minimum stable samples",
    )
    return v_tail, v_time_cap


@app.cell(hide_code=True)
def _(
    get_df,
    infer_shape,
    mo,
    np,
    plot_utils,
    plt,
    select_groups,
    v_angle_cap,
    v_angle_thr,
    v_angle_tol,
    v_baseline,
    v_compare,
    v_groups,
    v_horizon,
    v_omega_cap,
    v_omega_thr,
    v_omega_tol,
    v_panels,
    v_tail,
    v_time_cap,
):
    mo.stop(not v_groups.value, mo.callout("Select at least one group to display results.", kind="warn"))
    mo.stop(not v_panels.value, mo.callout("Select at least one plot panel to display.", kind="warn"))

    _horizon = v_horizon.value
    _labels = ["Earth", "Uranus"]
    _a_thr, _w_thr = v_angle_thr.value, v_omega_thr.value
    _a_tol, _w_tol, _tail = v_angle_tol.value, v_omega_tol.value, v_tail.value

    # ── Per-trajectory statistics for both datasets ──────────────────────────
    _stats, _frames = [], []
    for _body in ("earth", "uranus"):
        _df = select_groups(get_df(_body, _horizon), v_groups.value)
        _frames.append(_df)
        _batch, _n_steps = infer_shape(_df)
        _stats.append(
            plot_utils.create_stats_df(_df, _batch, _n_steps, _a_tol, _w_tol, _tail, labels=None)
        )
    _t_end = float(_frames[0]["Time"].max())

    def _summarise(frame):
        ang = frame["Final Angle Error"].to_numpy(dtype=float)
        om = frame["Final Omega Error"].to_numpy(dtype=float)
        slew = frame["Slew Time"].to_numpy(dtype=float)
        stable = ~np.isnan(slew)
        success = (ang < _a_thr) & (om < _w_thr) & stable
        return {
            "success": float(success.mean() * 100),
            "stable": float(stable.mean() * 100),
            "angle": float(ang.mean()),
            "omega": float(om.mean()),
            "slew": float(slew[stable].mean()) if stable.any() else float("nan"),
        }

    def _fmt(x, spec):
        return "—" if np.isnan(x) else format(x, spec)

    # ── Figures ──────────────────────────────────────────────────────────────
    try:
        _, _figs = plot_utils.plot_violin_and_bar(
            stats_df_list=_stats,
            angle_threshold=_a_thr,
            omega_threshold=_w_thr,
            time_hist_max=v_time_cap.value,
            angle_hist_max=v_angle_cap.value,
            omega_hist_max=v_omega_cap.value,
            dataset_labels=_labels,
            title=None,
            verbose=False,
            plot_bar_by_group=False,
            filename=None,
        )
    except Exception as _err:
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    f"Could not render the plots: `{_err}`  \n"
                    "If no trajectory meets the stability criteria, try larger tolerances or fewer "
                    "minimum stable samples."
                ),
                kind="danger",
            ),
        )

    _panel_defs = {
        "Slew time": ("Slew Times", _figs[0]),
        "Final angle error": ("Final Angle Error", _figs[1]),
        "Final angular velocity": ("Final Angular Velocity", _figs[2]),
        "Success rate": ("Success Rate", _figs[3]),
    }
    _panels = [
        mo.vstack([mo.md(f"**{_title}**"), _fig], align="center", gap=0.25)
        for _name, (_title, _fig) in _panel_defs.items()
        if _name in v_panels.value
    ]

    _pooled = {_lab: _summarise(_s) for _lab, _s in zip(_labels, _stats)}

    # ── Results table ────────────────────────────────────────────────────────
    _group_names = list(dict.fromkeys(g for s in _stats for g in s["Group"].unique()))
    _lines = [
        "| Group | Dataset | Success | Stable | Slew time (s) | Angle err. (deg) | Rate err. (deg/s) |",
        "|:--|:--|--:|--:|--:|--:|--:|",
    ]

    def _row(group, label, p, bold_success=False):
        success = f"{p['success']:.1f}%"
        if bold_success:
            success = f"**{success}**"
        return (
            f"| {group} | {label} | {success} | {p['stable']:.1f}% | {_fmt(p['slew'], '.1f')} "
            f"| {p['angle']:.3f} | {p['omega']:.3f} |"
        )

    for _g in _group_names:
        _per = {
            _lab: _summarise(_s.loc[_s["Group"] == _g])
            for _lab, _s in zip(_labels, _stats)
            if (_s["Group"] == _g).any()
        }
        _best = max(p["success"] for p in _per.values())
        _unique_best = sum(p["success"] == _best for p in _per.values()) == 1 and len(_per) > 1
        for _lab, _p in _per.items():
            _lines.append(_row(_g, _lab, _p, bold_success=_unique_best and _p["success"] == _best))

    # ── Group-comparison badges (one per planet) ─────────────────────────────
    _cmp, _base = v_compare.value, v_baseline.value

    _by_planet = {
        _lab: {_g: _summarise(_s.loc[_s["Group"] == _g]) for _g in _s["Group"].unique()}
        for _lab, _s in zip(_labels, _stats)
    }

    def _badge(label):
        c, b = _by_planet[label][_cmp]["success"], _by_planet[label][_base]["success"]
        delta = c - b
        caption = f"{_cmp} {c:.1f}% vs {_base} {b:.1f}%"
        return mo.stat(
            value=f"{delta:+.1f} %",
            label=f"{label} · {_cmp} vs {_base}",
            caption=caption,
            direction=None if delta == 0 else ("increase" if delta > 0 else "decrease"),
            bordered=True,
        )

    if _cmp == _base:
        _badges = mo.callout("Choose two different groups to compare.", kind="info")
    elif _cmp not in _group_names or _base not in _group_names:
        _badges = mo.callout(
            mo.md(f"Include both **{_cmp}** and **{_base}** in *Groups* to see their success-rate comparison."),
            kind="info",
        )
    else:
        _badges = mo.hstack([_badge(_lab) for _lab in _labels], justify="start", gap=1)

    _n_groups = len(_group_names)
    _title_md = mo.md(f"### Results: N = {_horizon} steps")
    _header_md = mo.md(
        f"**Success:** stable, with converged angle error < **{_a_thr:g}°** and "
        f"angular-velocity error < **{_w_thr:g}°/s**.  \n"
        f"**Stable:** final segment stays within **±{_a_tol:g}°** and **±{_w_tol:g}°/s** of the "
        f"tail means for at least **{_tail}** samples.  \n"
        f"{_n_groups} group{'s' if _n_groups != 1 else ''} × {_batch} trajectories per dataset."
    )

    _footnote = mo.md(
        "_Slew time, angle and rate columns are means. Badges show the success-rate difference between the two selected groups, in percentage points (pp), "
        "for each dataset. In the table, bold marks the higher success rate within a group. Axis limits affect the plots only; "
        "all statistics use the full trajectories. The aggregate “Full” group is excluded._"
    )

    _out = mo.vstack([
        _title_md, 
        _badges,
        _header_md,
        mo.hstack(_panels, justify="center", gap=1, wrap=True),
        mo.md("\n".join(_lines)),
        _footnote,
    ], gap=1.5)

    for _fig in _figs:
        plt.close(_fig)
    _out
    return


@app.cell(hide_code=True)
def _(
    control_grid,
    mo,
    v_angle_cap,
    v_angle_thr,
    v_angle_tol,
    v_baseline,
    v_compare,
    v_groups,
    v_horizon,
    v_omega_cap,
    v_omega_thr,
    v_omega_tol,
    v_panels,
    v_tail,
    v_time_cap,
    violin_reset_btn,
):
    _controls = mo.vstack([
        mo.md("**Data & display**"),
        control_grid([v_horizon, v_groups, v_panels]),
        mo.md("**Group comparison**"),
        control_grid([v_compare, v_baseline]),
        mo.md("**Success criteria**"),
        control_grid([v_angle_thr, v_omega_thr]),
        mo.md("**Stability detection**"),
        control_grid([v_angle_tol, v_omega_tol, v_tail]),
        mo.md("**Plot limits**"),
        control_grid([v_time_cap, v_angle_cap, v_omega_cap]),
        mo.hstack([violin_reset_btn], justify="end"),
    ], gap=1)

    mo.accordion({
        "Interact with data": _controls,
        "Definitions: stability, slew time, and success": mo.md(
            "A trajectory is **stable** if its final contiguous segment stays within the angle and "
            "rate tolerances of the *tail means* (averages over the last *minimum stable samples*) "
            "for at least that many samples. **Slew time** is the time at which that final segment "
            "begins. A trajectory is a **success** if it is stable *and* its tail-mean angle error "
            "and rate error are both below the success thresholds. Statistics always use the full "
            "trajectories, independent of the plot axis limits."
        ),
    }, multiple=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 2 · Error Trajectories: Earth vs. Uranus
    "
        "Angle and angular-velocity error histories. Groups are paged through in a carousel by "
        "default; overlay them to compare directly. Use **Interact with data** beneath the plots to "
        "change the view, thresholds and time window.
    """)
    return


@app.cell(hide_code=True)
def _(HORIZONS, get_cost_reset, get_df, mo):
    get_cost_reset()  # re-running this cell after a reset restores every default

    _default_h = 200 if 200 in HORIZONS else HORIZONS[-1]
    _groups = [g for g in get_df("earth", _default_h)["Group"].unique() if g != "Full"]
    _slider = dict(show_value=True, debounce=True, full_width=True)

    c_horizon = mo.ui.dropdown(
        options={f"N = {h}": h for h in HORIZONS}, value=f"N = {_default_h}",
        label="Horizon", full_width=True,
    )
    c_bodies = mo.ui.multiselect(
        options=["Earth", "Uranus"], value=["Earth", "Uranus"], label="Datasets", full_width=True,
    )
    c_groups = mo.ui.multiselect(options=_groups, value=_groups, label="Groups", full_width=True)

    c_compare = mo.ui.switch(value=True, label="Overlay groups on shared axes")
    c_stats = mo.ui.switch(value=True, label="Show summary statistics")
    c_traj_mode = mo.ui.dropdown(
        options=["All trajectories", "Single trajectory"], value="All trajectories",
        label="Trajectories", full_width=True,
    )

    c_angle_thr = mo.ui.slider(0.1, 90, value=15.0, step=0.1, label="Angle threshold (deg)", **_slider)
    c_omega_thr = mo.ui.slider(0.1, 30, value=5.0, step=0.1, label="Rate threshold (deg/s)", **_slider)
    c_show_thr = mo.ui.switch(value=True, label="Show threshold lines")
    c_fit_y = mo.ui.switch(value=True, label="Fit y-axis to time window")
    return (
        c_angle_thr,
        c_bodies,
        c_compare,
        c_fit_y,
        c_groups,
        c_horizon,
        c_omega_thr,
        c_show_thr,
        c_stats,
        c_traj_mode,
    )


@app.cell(hide_code=True)
def _(c_horizon, get_df, infer_shape, mo, np):
    # Controls whose range depends on the selected horizon
    _h = c_horizon.value
    _frames = [get_df(b, _h) for b in ("earth", "uranus")]
    _times = np.sort(_frames[0]["Time"].unique())
    _steps = list(dict.fromkeys(float(t) for t in np.round(_times, 3)))
    _batch = min(infer_shape(f)[0] for f in _frames)

    c_time = mo.ui.range_slider(
        steps=_steps, value=[_steps[0], _steps[-1]], show_value=True, debounce=True,
        full_width=True, label="Time window (s)",
    )
    c_traj_idx = mo.ui.slider(
        0, max(_batch - 1, 0), value=0, step=1, show_value=True, debounce=True,
        full_width=True, label="Trajectory index",
    )
    return c_time, c_traj_idx


@app.cell(hide_code=True)
def _(
    c_angle_thr,
    c_bodies,
    c_compare,
    c_fit_y,
    c_groups,
    c_horizon,
    c_omega_thr,
    c_show_thr,
    c_stats,
    c_time,
    c_traj_idx,
    c_traj_mode,
    cost_view,
    describe_view,
    get_df,
    infer_shape,
    mo,
    select_groups,
):
    mo.stop(not c_bodies.value, mo.callout("Select at least one dataset to display.", kind="warn"))
    mo.stop(not c_groups.value, mo.callout("Select at least one group to display.", kind="warn"))
    _t_lo, _t_hi = c_time.value
    mo.stop(_t_hi <= _t_lo, mo.callout("Choose a time window with a non-zero width.", kind="warn"))

    _horizon = c_horizon.value
    _single = c_traj_mode.value == "Single trajectory"
    _idx = int(c_traj_idx.value) if _single else None
    _bodies = [b for b in ("Earth", "Uranus") if b in c_bodies.value]
    _frames = {b: select_groups(get_df(b.lower(), _horizon), c_groups.value) for b in _bodies}
    _batch = min(infer_shape(f)[0] for f in _frames.values())
    _opts = dict(
        compare_groups=c_compare.value, batch_idx=_idx,
        angle_thr=c_angle_thr.value, omega_thr=c_omega_thr.value,
        t_window=(_t_lo, _t_hi), plot_stats=c_stats.value,
        show_thresholds=c_show_thr.value, fit_y=c_fit_y.value,
    )

    _n_groups = len(c_groups.value)
    _traj_text = f"trajectory #{_idx}" if _single else f"all {_batch} trajectories per group"
    _summary = mo.md(
        f"Showing **{' & '.join(_bodies)}** at horizon **N = {_horizon}** · "
        f"{_n_groups} group{'s' if _n_groups != 1 else ''} · {_traj_text} · "
        f"window **{_t_lo:g}–{_t_hi:g} s**  \n"
        f"_{describe_view(c_compare.value, c_stats.value)}_"
    )

    # Figures are only drawn once this section scrolls into view
    _view = mo.lazy(lambda: cost_view(_frames, label_sources=True, **_opts), show_loading_indicator=True)
    mo.vstack([_summary, _view], gap=1.5)
    return


@app.cell(hide_code=True)
def _(
    c_angle_thr,
    c_bodies,
    c_compare,
    c_fit_y,
    c_groups,
    c_horizon,
    c_omega_thr,
    c_show_thr,
    c_stats,
    c_time,
    c_traj_idx,
    c_traj_mode,
    control_grid,
    cost_reset_btn,
    mo,
):
    _traj_controls = [c_traj_mode]
    if c_traj_mode.value == "Single trajectory":
        _traj_controls.append(c_traj_idx)  # only shown when it has an effect

    mo.accordion({
        "Interact with data": mo.vstack([
            mo.md("**Data**"),
            control_grid([c_horizon, c_bodies, c_groups]),
            mo.md("**View**"),
            control_grid([c_compare, c_stats]),
            control_grid(_traj_controls),
            mo.md("**Thresholds and time window**"),
            control_grid([c_angle_thr, c_omega_thr]),
            control_grid([c_show_thr, c_fit_y]),
            control_grid([c_time], cols=1),
            mo.hstack([cost_reset_btn], justify="end"),
        ], gap=1)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 3 · Torque vs. Dipole
    "
        "Angle and angular-velocity error histories for the torque comparison dataset. "
        "Use **Interact with data** beneath the plots to change the view, thresholds and time window.
    """)
    return


@app.cell(hide_code=True)
def _(get_torque, get_torque_reset, infer_shape, mo, np):
    get_torque_reset()  # re-running this cell after a reset restores every default

    _df = get_torque()
    _groups = list(_df["Group"].unique())
    _batch, _ = infer_shape(_df)
    _times = np.sort(_df["Time"].unique())
    _steps = list(dict.fromkeys(float(t) for t in np.round(_times, 3)))
    _hi = max([t for t in _steps if t <= 80.0], default=_steps[-1])  # original default: 80 s
    _slider = dict(show_value=True, debounce=True, full_width=True)

    tq_groups = mo.ui.multiselect(options=_groups, value=_groups, label="Groups", full_width=True)
    tq_compare = mo.ui.switch(value=True, label="Overlay groups on shared axes")
    tq_stats = mo.ui.switch(value=False, label="Show summary statistics")
    tq_traj_mode = mo.ui.dropdown(
        options=["All trajectories", "Single trajectory"], value="All trajectories",
        label="Trajectories", full_width=True,
    )
    tq_traj_idx = mo.ui.slider(
        0, max(_batch - 1, 0), value=0, step=1, label="Trajectory index", **_slider,
    )

    tq_angle_thr = mo.ui.slider(0.1, 90, value=15.0, step=0.1, label="Angle threshold (deg)", **_slider)
    tq_omega_thr = mo.ui.slider(0.1, 30, value=5.0, step=0.1, label="Rate threshold (deg/s)", **_slider)
    tq_show_thr = mo.ui.switch(value=True, label="Show threshold lines")
    tq_fit_y = mo.ui.switch(value=True, label="Fit y-axis to time window")
    tq_time = mo.ui.range_slider(
        steps=_steps, value=[_steps[0], _hi], label="Time window (s)", **_slider,
    )
    return (
        tq_angle_thr,
        tq_compare,
        tq_fit_y,
        tq_groups,
        tq_omega_thr,
        tq_show_thr,
        tq_stats,
        tq_time,
        tq_traj_idx,
        tq_traj_mode,
    )


@app.cell(hide_code=True)
def _(
    cost_view,
    describe_view,
    get_torque,
    infer_shape,
    mo,
    select_groups,
    tq_angle_thr,
    tq_compare,
    tq_fit_y,
    tq_groups,
    tq_omega_thr,
    tq_show_thr,
    tq_stats,
    tq_time,
    tq_traj_idx,
    tq_traj_mode,
):
    mo.stop(not tq_groups.value, mo.callout("Select at least one group to display.", kind="warn"))
    _t_lo, _t_hi = tq_time.value
    mo.stop(_t_hi <= _t_lo, mo.callout("Choose a time window with a non-zero width.", kind="warn"))

    _df = select_groups(get_torque(), tq_groups.value)
    _batch, _ = infer_shape(_df)
    _single = tq_traj_mode.value == "Single trajectory"
    _idx = int(tq_traj_idx.value) if _single else None
    _opts = dict(
        compare_groups=tq_compare.value, batch_idx=_idx,
        angle_thr=tq_angle_thr.value, omega_thr=tq_omega_thr.value,
        t_window=(_t_lo, _t_hi), plot_stats=tq_stats.value,
        show_thresholds=tq_show_thr.value, fit_y=tq_fit_y.value,
    )

    _names = ", ".join(str(g) for g in tq_groups.value)
    _traj_text = f"trajectory #{_idx}" if _single else f"all {_batch} trajectories per group"
    _summary = mo.md(
        f"Showing **{_names}** · {_traj_text} · window **{_t_lo:g}–{_t_hi:g} s**  \n"
        f"_{describe_view(tq_compare.value, tq_stats.value)}_"
    )

    # Figures are only drawn once this section scrolls into view
    _view = mo.lazy(
        lambda: cost_view({"Torque": _df}, label_sources=False, **_opts), show_loading_indicator=True
    )
    mo.vstack([_summary, _view], gap=1.5)
    return


@app.cell(hide_code=True)
def _(
    control_grid,
    mo,
    torque_reset_btn,
    tq_angle_thr,
    tq_compare,
    tq_fit_y,
    tq_groups,
    tq_omega_thr,
    tq_show_thr,
    tq_stats,
    tq_time,
    tq_traj_idx,
    tq_traj_mode,
):
    _traj_controls = [tq_traj_mode]
    if tq_traj_mode.value == "Single trajectory":
        _traj_controls.append(tq_traj_idx)  # only shown when it has an effect

    mo.accordion({
        "Interact with data": mo.vstack([
            mo.md("**Data**"),
            control_grid([tq_groups], cols=1),
            mo.md("**View**"),
            control_grid([tq_compare, tq_stats]),
            control_grid(_traj_controls),
            mo.md("**Thresholds and time window**"),
            control_grid([tq_angle_thr, tq_omega_thr]),
            control_grid([tq_show_thr, tq_fit_y]),
            control_grid([tq_time], cols=1),
            mo.hstack([torque_reset_btn], justify="end"),
        ], gap=1)
    })
    return


if __name__ == "__main__":
    app.run()
