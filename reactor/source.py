"""Source track construction shared by AMASS preprocessing and new (e.g. seamless) clips.

A *track* holds, at 50 Hz, the global frames of the SMPL+H joints that have a robot
correspondence, plus the per-motion constants of the nominal mapping: scale s, x_nom
(Eq. 8) and the vertical offset z_nom. ``R_nom`` (Eq. 9) is shared by all motions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from reactor.robot import RobotInfo, build_model
from reactor.rotations import mat_to_quat, quat_to_rotvec
from reactor.smplh import CANON_TO_WORLD, NUM_BODY_JOINTS, _model, forward_kinematics

FPS = 50.0


@dataclass
class TargetContext:
    robot: RobotInfo
    tgt_pos: np.ndarray  # nominal target body positions (B, 3)
    h_target: float
    smpl_ids: np.ndarray
    R_nom: np.ndarray  # (B, 3, 3), Eq. 9
    clearance: torch.Tensor  # per-body ground clearance used for z_nom

    @classmethod
    def create(cls, device: torch.device, robot: str = "g1") -> TargetContext:
        robot = RobotInfo(build_model(robot), robot)
        tgt_pos, tgt_rot = robot.nominal_frames()
        # The source nominal configuration is the zero pose with root orientation C, so every
        # source global frame equals C and R_nom = C^T R_target (Eq. 9).
        R_nom = (CANON_TO_WORLD.T @ tgt_rot).astype(np.float32)
        clearance = torch.full((len(robot.smpl_ids),), 0.05, device=device)
        for fb in robot.foot_ids:
            clearance[list(robot.body_ids).index(fb)] = robot.spec.foot_clearance
        return cls(robot, tgt_pos, float(tgt_pos[0, 2]), robot.smpl_ids, R_nom, clearance)


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


def body_shape(gender: str, betas: np.ndarray) -> tuple[float, np.ndarray]:
    """Pelvis height above the soles and canonical rest joints for a subject."""
    v_template, shapedirs, J_reg = _model(gender)
    nb = min(len(betas), shapedirs.shape[-1])
    v = v_template + shapedirs[..., :nb] @ betas[:nb]
    rest = J_reg @ v
    return float(rest[0, 1] - v[:, 1].min()), rest


def make_track(ctx: TargetContext, rest: np.ndarray, h_src: float, trans: np.ndarray, aa: np.ndarray,
               src_fps: float, device: torch.device) -> dict:
    """Build one source track. ``trans`` (T, 3) and ``aa`` (T, 22, 3) are in a Z-up world."""
    s = ctx.h_target / h_src
    trans_t = torch.tensor(trans, dtype=torch.float64, device=device)
    aa_t = torch.tensor(aa[:, :NUM_BODY_JOINTS], dtype=torch.float64, device=device)
    trans_t, aa_t = resample(trans_t, aa_t, src_fps)
    rest_t = torch.tensor(rest, dtype=torch.float64, device=device)
    pos, rot = forward_kinematics(rest_t, trans_t, aa_t)  # (n, 22, 3), (n, 22, 3, 3)

    # Eq. 8: source nominal frames = zero pose, root orientation C, soles at z = 0.
    C = torch.tensor(CANON_TO_WORLD, dtype=torch.float64, device=device)
    src_nom = (C @ rest_t.T).T
    src_nom = src_nom - src_nom[0] + torch.tensor([0, 0, h_src], device=device, dtype=torch.float64)
    src_nom = src_nom[ctx.smpl_ids].cpu().numpy()
    x_nom = np.einsum("ij,bi->bj", CANON_TO_WORLD, ctx.tgt_pos - s * src_nom)  # C^T (x_t - s x_s)

    # Per-motion nominal vertical offset z_nom: lowest mapped body touches the ground.
    Rm = rot[:, ctx.smpl_ids].float()
    xm = pos[:, ctx.smpl_ids].float()
    xg = (Rm @ torch.tensor(x_nom, device=device).float().unsqueeze(-1)).squeeze(-1) + s * xm
    # 98th percentile of the per-frame lift needed so a few noisy frames do not lift
    # the whole motion.
    z_nom = float((ctx.clearance - xg[..., 2]).max(-1).values.quantile(0.98).item())

    # Source foot contacts (Shimada et al. 2020 style): low and slow ankle/toe joints.
    feet = pos[:, [7, 10, 8, 11]].float()  # L_Ankle, L_Foot, R_Ankle, R_Foot
    floor = feet[..., 2].reshape(-1).quantile(0.02)
    vel = torch.zeros_like(feet)
    vel[1:] = (feet[1:] - feet[:-1]) * FPS
    low = (feet[..., 2] - floor) < torch.tensor([0.12, 0.06, 0.12, 0.06], device=device)
    slow = vel.norm(dim=-1) < 0.4
    c = low & slow
    contact = torch.stack([c[:, 0] | c[:, 1], c[:, 2] | c[:, 3]], -1)

    return {
        "pos": xm.cpu().numpy().astype(np.float32),
        "quat": mat_to_quat(Rm).cpu().numpy().astype(np.float32),
        "contact": contact.cpu().numpy(),
        "scale": s,
        "x_nom": x_nom.astype(np.float32),
        "z_nom": z_nom,
    }


def save_tracks(path, ctx: TargetContext, tracks: list[dict], names: list[str]) -> int:
    lengths = np.array([len(t["pos"]) for t in tracks], np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        pos=np.concatenate([t["pos"] for t in tracks]),
        quat=np.concatenate([t["quat"] for t in tracks]),
        contact=np.concatenate([t["contact"] for t in tracks]),
        starts=np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64),
        lengths=lengths,
        scale=np.array([t["scale"] for t in tracks], np.float32),
        x_nom=np.stack([t["x_nom"] for t in tracks]),
        z_nom=np.array([t["z_nom"] for t in tracks], np.float32),
        R_nom=ctx.R_nom,
        robot=ctx.robot.name,
        names=np.array(names),
        fps=FPS,
    )
    return int(lengths.sum())
