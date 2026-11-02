# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "raddo[cuda13]",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="Intro to RADDO")

with app.setup(hide_code=True):
    import marimo as mo

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

    # Jax configuration
    jax.config.update("jax_default_matmul_precision", "highest") # required for vmap to work properly, weird bug
    jax.config.update("jax_enable_x64", True)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Introduction to RApid Data-Driven Optimization (RADDO)
    ## What is RADDO?
    RADDO is a JAX-based, GPU-accelerated library for trajectory generation, model predictive control, and online-adaptation.
    It is designed to be easily used for your application. You can run it on a marimo server (make sure to enable GPU).
    ### Features
    - [Just-in-time (JIT) compilation](https://docs.jax.dev/en/latest/jit-compilation.html)
     - Makes use of the [Equinox](https://docs.kidger.site/equinox/) library for Object-Oriented programming while maintaining JIT-compatibility
    - Quadratic Program (QP) Controller built on [Moreau](https://docs.moreau.so/)
      - Differentiable, fast QP solves
    - Attitude-Aware QP Solver
     - The QP Solver is set up to natively handle quaternion attitude dynamics
       - Based off of ["Planning with Attitude"](https://roboticexplorationlab.org/papers/planning_with_attitude.pdf)
    - SQP Controller
    - Sampling-based MPC controllers (MPPI/CEM)
    - Quick start
     - [marimo](https://marimo.io/) notebooks offer their own [sandboxed environments](https://docs.marimo.io/guides/package_management/sandboxes/), so you don't need to install any of your own dependencies
       - Just open this notebook `uvx marimo edit --sandbox Intro_to_RADDO.py`
       - You can also open it in [molab](https://molab.marimo.io/) which provides GPU access
     - RADDO is also a public PyPi package, so you can also get started with `pip install raddo` or `pip install raddo[cuda13]`
    - Modular design
     - It is designed to be as general as possible so that you can use it for your own project.
     - You can use your own dynamics and/or write your own controllers.
    - Differentiable
     - The whole pipeline is differentiable similar to [TurboMPC](https://arxiv.org/pdf/2606.24039) and [DiffMPC](https://github.com/ToyotaResearchInstitute/diffmpc).
     -  This allows you to use gradient descent to learn better cost weights, dynamics parameters, etc
        - This is not something we have experimented with yet.
     - The structure of the Dynamics class is heavily based on [DiffMPC](https://github.com/ToyotaResearchInstitute/diffmpc).

    ### This notebook
    This notebook walks you through the basic functionality of the library, demonstrating how you could apply it to your own problem. If you want a more comprehensive example associated with the paper ["Learning-Based Adaptive MPC for Magnetorquer-Only Satellite Attitude Control in an Unknown Magnetic Field"](https://arxiv.org/abs/2610.20280), see the `Magnetorquer-Control.py` notebook [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/github/github.com/JHU-ACEL/raddo/blob/main/notebooks/Magnetorquer_Control.py).

    ### marimo notes
    If you have never worked with marimo before, it is a reactive python notebook, which means all the cells run automatically. Set expensive cells to run on a button-click, or disable autorun.
    Also, you cannot redefine variables. If a variable is only needed in that cell (e.g. fig), define it as (_fig)

    ------------------------------------------------------------------------------------
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Check what devices are available. In molab, make sure to enable GPU.
    """)
    return


@app.cell
def _():
    jax.devices()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Example with included dynamics and controllers
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Define the system
    """)
    return


@app.cell
def _():
    from raddo.dynamics import DoubleIntegratorDynamics
    from raddo.utils import TrajectoryGenerator
    dt = 0.1
    dynamics = DoubleIntegratorDynamics()
    system = TrajectoryGenerator(dynamics=dynamics, dt=dt)
    return dynamics, system


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Propagate initial conditions
    """)
    return


@app.cell
def _(system):
    N = 100 # timesteps
    x0 = jnp.array([10., 6., 1., 0.]) # x,y,vx,vy
    key = jrandom.key(123)

    _trajectory, _controls = system.generate_trajectory(initial_state=x0, key=key, num_steps=N)
    system.plot_traj(_trajectory, _controls)
    return N, key, x0


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Propagate fixed control sequence
    """)
    return


@app.cell
def _(N, key, system, x0):
    control_sequence = 0.5*jrandom.normal(key, shape=(N, 2)) # Controls are Fx, Fu
    _traj, _cntrl = system.generate_trajectory(initial_state=x0, key=key, num_steps=N, control_sequence=control_sequence)
    system.plot_traj(_traj, _cntrl)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Generate batch of trajectories with noise. This function works with different initial states, different control sequences, etc (as long as the batch dimension of all of them matches)
    """)
    return


@app.cell
def _(key, system, x0):
    _trajs, _cntrls = system.generate_trajectory_batch(initial_states=x0, key=key,noise_std=1, batch_size=100)
    system.plot_traj(_trajs)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now let's control a trajectory
    """)
    return


@app.cell
def _(N, dynamics, key, system):
    from raddo.controllers import QPController

    # Weight matrices
    Q = jnp.eye(dynamics.num_states)
    Qf = Q
    R = jnp.eye(dynamics.num_controls)

    # Planning horizon of QP Solver
    horizon_qp = 50

    state_limits = jnp.array([[-20, 20]]*2 + [[-2,2]]*2)
    control_limits = jnp.array([[-2, 2]] * dynamics.num_controls) 

    qp_controller = QPController(system, Q, Qf, R, state_limits, control_limits, horizon_qp)

    # Batch of initial states
    batch_size = 100
    _, state_key, traj_key = jrandom.split(key,3)
    x0_batch = 0.5*jrandom.normal(state_key, shape=(batch_size, 4)) 

    # Target
    x_goal = jnp.zeros(4)

    # Generate trajectories
    _trajs, _cntrls = system.generate_trajectory_batch(initial_states=x0_batch, key=traj_key, target_states=x_goal, noise_std=1e-2, high_level_controller=qp_controller, num_steps=N, batch_size=batch_size)
    system.plot_traj(_trajs)
    return


if __name__ == "__main__":
    app.run()
