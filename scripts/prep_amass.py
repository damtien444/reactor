"""Preprocess AMASS (SMPL+H G) clips into source-frame tracks for ReActor.

For every kept clip we store, at 50 Hz, the global position and orientation of the SMPL+H
joints that have a robot correspondence (``RobotSpec.correspondences`` in reactor.robot), the per-motion
scale s and nominal offsets x_nom (Eq. 8; these depend on the subject's betas), and an
estimated source foot-contact signal used only for evaluation.

Filtering approximates the PHC criteria: clips with human-object/scene interaction
(chairs, stairs, tables, ...) are removed by name, as are very short clips.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch

from reactor import DATA
from reactor.robot import ROBOTS
from reactor.source import TargetContext, body_shape, make_track, save_tracks

MIN_SECONDS = 1.0
MAX_SECONDS = 60.0

# Clip-name keywords indicating object/scene interaction or unsupported support surfaces.
EXCLUDE = re.compile(
    r"chair|sit|seat|stool|table|bench|bed|sofa|couch|stair|step_over|stepover|step_stone|"
    r"box|ladder|beam|handrail|seesaw|slope|ramp|climb|lean|wall|door|car|bike|bicycle|"
    r"horse|treadmill|rope|hang|swing|pole|bar_|monkey|platform|obstacle|pizza|tray|"
    r"suitcase|bag|cup|bottle|kettle|knife|cut|ball|basket|racket|bat|golf|tennis|guitar|"
    r"violin|drum|piano|mix|pour|stir|drink|wipe|shower|dry_|sponge|broom|vacuum|shovel|"
    r"hammer|saw|screw|rake|pick|carry|lift|push_from|pull|drag|object|lie|lying|crawl_under|"
    r"jump_on|jump_off|hop_on|gap|parkour",
    re.IGNORECASE,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amass", type=Path, default=DATA / "amass")
    ap.add_argument("--robot", choices=list(ROBOTS), default="g1")
    ap.add_argument("--out", type=Path, help="default: data/processed/amass_<robot>.npz")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    args.out = args.out or DATA / "processed" / f"amass_{args.robot}.npz"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = TargetContext.create(dev, args.robot)

    files = sorted(args.amass.rglob("*_poses.npz"))
    kept = [f for f in files if not EXCLUDE.search(str(f.relative_to(args.amass)))]
    if args.limit:
        kept = kept[: args.limit]
    print(f"{len(files)} clips, {len(kept)} after name filter")

    tracks, names = [], []
    for k, f in enumerate(kept):
        name = str(f.relative_to(args.amass))
        d = np.load(f)
        src_fps = float(d["mocap_framerate"])
        T = d["poses"].shape[0]
        if T / src_fps < MIN_SECONDS:
            continue
        T = min(T, int(MAX_SECONDS * src_fps))
        h_src, rest = body_shape(str(d["gender"]), d["betas"].astype(np.float64))
        aa = d["poses"][:T, :66].reshape(T, 22, 3)
        track = make_track(ctx, rest, h_src, d["trans"][:T], aa, src_fps, dev)
        tracks.append(track)
        names.append(name)
        if k % 250 == 0:
            print(f"[{k}/{len(kept)}] {name} frames={len(track['pos'])} s={track['scale']:.3f} "
                  f"z_nom={track['z_nom']:.3f}", flush=True)

    frames = save_tracks(args.out, ctx, tracks, names)
    print(f"saved {len(tracks)} motions, {frames} frames ({frames / 50 / 3600:.2f} h) -> {args.out}")


if __name__ == "__main__":
    main()
