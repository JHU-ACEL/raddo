# Rapidly Adapting Data-Drive Optimization (RADDO) 

RADDO is a JAX-based, GPU-accelerated library for trajectory generation, model predictive control, and online-adaptation.
This library was built for [Learning-Based Adaptive MPC for Magnetorquer-Only Satellite Attitude Control](https://arxiv.org/abs/2610.20280) which was accepted to the International Conference on Space Robotics (iSpaRo) 2026.

## Learning-Based Adaptive MPC for Magnetorquer-Only Satellite Attitude Control
<img width="6832" height="4143" alt="hero" src="https://github.com/user-attachments/assets/b51ff63d-21c4-4f67-a356-42faa18d1b24" />

## Quick start
Explore our example notebook [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/JHU-ACEL/raddo/blob/main/notebooks/Intro_to_RADDO.py).

[Marimo](https://marimo.io/) offers a sandboxed environment with access to a GPU, so you will not need to install anything. If you do not want to run it in [molab](https://molab.marimo.io/notebooks), you can just run the following locally
```
uvx marimo edit --sandbox Intro_to_RADDO.py
```

## Other ways to work with library
1. pip install into your own environment
```
pip install raddo # or raddo[cuda13]
```
2. Pull our pre-built Docker images
```
docker pull ghcr.io/jhu-acel/raddo:latest
docker pull ghcr.io/jhu-acel/raddo:dev # optional: editable pip install to make changes
```
3. Clone this repository
```
git clone https://github.com/JHU-ACEL/raddo
```
and then run 
```
docker compose up cli # base install
docker compose up dev # editable pip install
```
Then open the marimo notebook that started by going to `localhost:8888`.
To attach to the docker container command line, run 
```
docker exec -it -u docker raddo-cli-1 bash
```

## Examples
- `Intro_to_RADDO.py` demonstrates the basic functionality of the library [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/JHU-ACEL/raddo/blob/main/notebooks/Intro_to_RADDO.py)
- `Magnetorquer_Control.py` is a more comprehensive example for the magnetorquer-only attitude control problem of a satellite with online adaptation [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/JHU-ACEL/raddo/blob/main/notebooks/Magnetorquer_Control.py)
- `B_field_Pretraining.py` shows the training pipeline for training the magnetic field models offline. [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/JHU-ACEL/raddo/blob/main/notebooks/B_Field_Pretraining.py)

## Details
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
   - To get started run `uvx marimo edit --sandbox Intro_to_RADDO.py`
   - You can also open it in [molab](https://molab.marimo.io/) which provides GPU access
- Modular design
 - It is designed to be as general as possible so that you can use it for your own project.
 - You can use your own dynamics and/or write your own controllers.
- Differentiable
 - The whole pipeline is differentiable similar to [TurboMPC](https://arxiv.org/pdf/2606.24039) and [DiffMPC](https://github.com/ToyotaResearchInstitute/diffmpc).
 -  This allows you to use gradient descent to learn better cost weights, dynamics parameters, etc
    - This is not something we have experimented with yet.
 - The structure of the Dynamics class is heavily based on [DiffMPC](https://github.com/ToyotaResearchInstitute/diffmpc).

## Cite
```
@article{SchwartzCauligi2026,
  title={Learning-Based Adaptive MPC for Magnetorquer-Only Satellite Attitude Control}, 
  author={Patrick Schwartz and Abhishek Cauligi},
  journal={International Conference on Space Robotics 2026},
  year={2026},
  url={https://arxiv.org/abs/2610.20280}
```
