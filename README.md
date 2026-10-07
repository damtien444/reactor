# ReActor reproduction (arXiv 2605.06593)

Reimplementation of *ReActor: Reinforcement Learning for Physics-Aware Motion Retargeting*
(Müller et al., SIGGRAPH 2026) for the Unitree G1 on AMASS.

## Layout

| File | Paper |
|---|---|
| `reactor/robot.py` | G1 model (from holosoma), PD gains, nominal T-pose, sparse correspondences (Fig. 4) |
| `reactor/smplh.py` | SMPL+H skeleton FK (body joints) |
| `scripts/prep_amass.py` | AMASS → source frames at 50 Hz, scale `s`, `x_nom`, `R_nom` (Eqs. 8–9), `z_nom`, source contacts |
| `reactor/motion.py` | Parameterized mapping `g_t` (Eqs. 10–13), projection onto `P` (Eq. 14) |
| `reactor/env.py` | Lower-level RL environment (Sec. 6): PD + residual root wrench with deadband (Eqs. 18–19), observations (Eq. 20), retargeting phase ψ, adaptive sampling, rewards (Tab. 1) |
| `reactor/ppo.py` | PPO with adaptive LR, MLP 3×512 ELU (Tab. 2) |
| `reactor/bilevel.py` | Upper level: TTSA single-loop update with simplified gradient (Eqs. 3, 6, 7, 15–17) |
| `scripts/train.py` | Joint training; `--no-bilevel` gives the RL-only ablation |
| `scripts/evaluate.py` | Table 3/4 kinematic metrics + export of retargeted trajectories |
| `scripts/plot_training.py` | Fig. 6 curves |

## Usage

```bash
uv sync
export PYTHONPATH=$PWD
uv run python scripts/prep_amass.py                       # -> data/processed/amass_g1.npz
uv run python scripts/train.py --run reactor_g1           # 20k iterations
uv run python scripts/train.py --run rlonly_g1 --no-bilevel
uv run python scripts/evaluate.py --run reactor_g1
uv run python scripts/plot_training.py
```

## Deviations from the paper

- **Simulator:** MuJoCo Warp instead of Isaac Sim (uv-managed env). 4096 envs, 200 Hz physics, 50 Hz control.
- **Data:** AMASS subsets ACCAD + CMU + KIT (15.7 h after filtering) instead of the full PHC-filtered AMASS.
  The PHC filter is approximated by excluding clips whose names indicate object/scene interaction.
- **Upper-level step size:** η = 1e-4 (the paper does not report η). With 1e-2 the parameters hit the
  bounds of `P` within a few hundred iterations.
- **Unspecified details chosen here:** wrench scales (100 N / 20 Nm per action unit), termination thresholds
  for adaptive sampling (0.5 m root, 45° torso), ψ ramp of 1 s, warm-start of 1000 iterations,
  two future reference frames in the observation, source-contact heuristic (height + velocity thresholds).
- **Not reproduced:** GMR / OmniRetarget baselines, the downstream tracking policy (Tab. 4 right part),
  Lima and ANYmal targets, and hardware experiments.
