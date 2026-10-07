"""Fig. 6: tracking reward, upper-level loss and parameter update rate, ReActor vs. RL only."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from reactor import ROOT


def load(run, tag):
    ea = EventAccumulator(str(ROOT / "runs" / run), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return [], []
    ev = ea.Scalars(tag)
    return [e.step for e in ev], [e.value for e in ev]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["reactor_g1", "rlonly_g1"])
    ap.add_argument("--labels", nargs="+", default=["ReActor", "RL only"])
    ap.add_argument("--warmstart", type=int, default=1000)
    args = ap.parse_args()
    tags = [("reward/tracking", r"$r_\mathrm{tracking}$"), ("upper/upper_loss", r"$\mathcal{L}$"),
            ("upper/param_update", r"$\|p_{k+1}-p_k\|_2$")]
    fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
    for run, label in zip(args.runs, args.labels):
        for ax, (tag, _) in zip(axes, tags):
            x, y = load(run, tag)
            ax.plot(x, y, label=label, lw=1)
    for ax, (_, name) in zip(axes, tags):
        ax.set_ylabel(name)
        ax.axvline(args.warmstart, ls="--", c="gray", lw=0.8)
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("Training steps")
    fig.tight_layout()
    path = ROOT / "runs" / "fig6_training_curves.png"
    fig.savefig(path, dpi=150)
    print(path)


if __name__ == "__main__":
    main()
