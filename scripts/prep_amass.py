"""Preprocess AMASS (SMPL+H G) clips into source-frame tracks for ReActor.

For every kept clip we store, at 50 Hz, the global position and orientation of the SMPL+H
joints that have a robot correspondence (reactor.robot.CORRESPONDENCES), the per-motion
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
from reactor.robot import CORRESPONDENCES, NOMINAL_JOINTS, RobotInfo, build_model
from reactor.rotations import exp_map, mat_to_quat, quat_to_mat, quat_to_rotvec
from reactor.smplh import CANON_TO_WORLD, NUM_BODY_JOINTS, _model, forward_kinematics, rest_joints

FPS = 50.0
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


def aa_to_quat(aa: torch.Tensor) -> torch.Tensor:
    angle = aa.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    s = torch.where(angle < 1e-8, 0.5 - angle**2 / 48.0, torch.sin(half) / angle.clamp_min(1e-12))
    return torch.cat([torch.cos(half), aa * s], -1)


def resample(trans: torch.Tensor, aa: torch.Tensor, src_fps: float):
    """Linear (trans) / normalized-lerp (local quats) resampling to FPS."""
    T = trans.shape[0]
    duration = (T - 1) / src_fps
    n = int(np.floor(duration * FPS)) + 1
    t = torch.arange(n, dtype=trans.dtype, device=trans.device) * (src_fps / FPS)
    i0 = t.floor().long().clamp(max=T - 1)
    i1 = (i0 + 1).clamp(max=T - 1)
    w = (t - i0).unsqueeze(-1)
    trans_r = trans[i0] * (1 - w) + trans[i1] * w
    q = aa_to_quat(aa)
    q0, q1 = q[i0], q[i1]
    sign = torch.where((q0 * q1).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    w = w.unsqueeze(-1)
    qr = q0 * (1 - w) + sign * q1 * w
    qr = qr / qr.norm(dim=-1, keepdim=True)
    return trans_r, quat_to_rotvec(qr)


def source_pelvis_height(gender: str, betas: np.ndarray) -> tuple[float, np.ndarray]:
    v_template, shapedirs, J_reg = _model(gender)
    nb = min(len(betas), shapedirs.shape[-1])
    v = v_template + shapedirs[..., :nb] @ betas[:nb]
    rest = J_reg @ v
    return float(rest[0, 1] - v[:, 1].min()), rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amass", type=Path, default=DATA / "amass")
    ap.add_argument("--out", type=Path, default=DATA / "processed" / "amass_g1.npz")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    robot = RobotInfo(build_model())
    tgt_pos, tgt_rot = robot.nominal_frames()
    h_target = float(tgt_pos[0, 2])
    smpl_ids = robot.smpl_ids
    C = torch.tensor(CANON_TO_WORLD, dtype=torch.float64, device=dev)
    # The source nominal configuration is the zero-pose with root orientation C, so every
    # source global frame equals C and R_nom = C^T R_target (Eq. 9).
    R_nom = (C.T.cpu().numpy() @ tgt_rot).astype(np.float32)

    files = sorted(args.amass.rglob("*_poses.npz"))
    kept, names = [], []
    for f in files:
        rel = str(f.relative_to(args.amass))
        if EXCLUDE.search(rel):
            continue
        kept.append(f)
        names.append(rel)
    if args.limit:
        kept, names = kept[: args.limit], names[: args.limit]
    print(f"{len(files)} clips, {len(kept)} after name filter")

    pos_all, quat_all, contact_all = [], [], []
    starts, lengths, scales, x_noms, z_noms, out_names = [], [], [], [], [], []
    offset = 0
    for k, (f, name) in enumerate(zip(kept, names)):
        d = np.load(f)
        src_fps = float(d["mocap_framerate"])
        T = d["poses"].shape[0]
        if T / src_fps < MIN_SECONDS:
            continue
        T = min(T, int(MAX_SECONDS * src_fps))
        gender = str(d["gender"])
        betas = d["betas"].astype(np.float64)
        h_src, rest = source_pelvis_height(gender, betas)
        s = h_target / h_src

        trans = torch.tensor(d["trans"][:T], dtype=torch.float64, device=dev)
        aa = torch.tensor(d["poses"][:T, : 3 * NUM_BODY_JOINTS], dtype=torch.float64, device=dev)
        trans, aa = resample(trans, aa.reshape(T, NUM_BODY_JOINTS, 3), src_fps)
        rest_t = torch.tensor(rest, dtype=torch.float64, device=dev)
        pos, rot = forward_kinematics(rest_t, trans, aa)  # (n, 22, 3), (n, 22, 3, 3)

        # Eq. 8: source nominal frames = zero pose, root orientation C, soles at z = 0.
        src_nom = (C @ rest_t.T).T
        src_nom = src_nom - src_nom[0] + torch.tensor([0, 0, h_src], device=dev, dtype=torch.float64)
        src_nom = src_nom[smpl_ids].cpu().numpy()
        x_nom = np.einsum("ij,bi->bj", CANON_TO_WORLD, tgt_pos - s * src_nom)  # C^T (x_t - s x_s)

        # Per-motion nominal vertical offset z_nom: lowest mapped body touches the ground.
        Rm = rot[:, smpl_ids].float()
        xm = pos[:, smpl_ids].float()
        xg = (Rm @ torch.tensor(x_nom, device=dev).float().unsqueeze(-1)).squeeze(-1) + s * xm
        clearance = torch.full((len(smpl_ids),), 0.05, device=dev)
        for fb in robot.foot_ids:
            clearance[list(robot.body_ids).index(fb)] = 0.035
        # 98th percentile of the per-frame lift needed so a few noisy frames do not lift
        # the whole motion.
        z_nom = float((clearance - xg[..., 2]).max(-1).values.quantile(0.98).item())

        # Source foot contacts (Shimada et al. 2020 style): low and slow ankle/toe joints.
        feet = pos[:, [7, 10, 8, 11]].float()  # L_Ankle, L_Foot, R_Ankle, R_Foot
        floor = feet[..., 2].reshape(-1).quantile(0.02)
        vel = torch.zeros_like(feet)
        vel[1:] = (feet[1:] - feet[:-1]) * FPS
        low = (feet[..., 2] - floor) < torch.tensor([0.12, 0.06, 0.12, 0.06], device=dev)
        slow = vel.norm(dim=-1) < 0.4
        c = low & slow
        contact = torch.stack([c[:, 0] | c[:, 1], c[:, 2] | c[:, 3]], -1)

        n = pos.shape[0]
        pos_all.append(xm.cpu().numpy().astype(np.float32))
        quat_all.append(mat_to_quat(Rm).cpu().numpy().astype(np.float32))
        contact_all.append(contact.cpu().numpy())
        starts.append(offset)
        lengths.append(n)
        scales.append(s)
        x_noms.append(x_nom.astype(np.float32))
        z_noms.append(z_nom)
        out_names.append(name)
        offset += n
        if k % 250 == 0:
            print(f"[{k}/{len(kept)}] {name} frames={n} s={s:.3f} z_nom={z_nom:.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        pos=np.concatenate(pos_all),
        quat=np.concatenate(quat_all),
        contact=np.concatenate(contact_all),
        starts=np.array(starts, np.int64),
        lengths=np.array(lengths, np.int64),
        scale=np.array(scales, np.float32),
        x_nom=np.stack(x_noms),
        z_nom=np.array(z_noms, np.float32),
        R_nom=R_nom,
        names=np.array(out_names),
        fps=FPS,
    )
    hours = offset / FPS / 3600
    print(f"saved {len(starts)} motions, {offset} frames ({hours:.2f} h) -> {args.out}")


if __name__ == "__main__":
    main()
