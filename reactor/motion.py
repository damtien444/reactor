"""Source motion library and the parameterized source -> target mapping (Sec. 5, Eqs. 10-14)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from reactor.rotations import exp_map, quat_to_mat


class MotionLib:
    """All source frames of a dataset concatenated on the GPU."""

    def __init__(self, path: Path, device: torch.device, names_filter=None):
        d = np.load(path)
        keep = np.arange(len(d["starts"]))
        if names_filter is not None:
            keep = np.array([i for i, n in enumerate(d["names"]) if names_filter(str(n))], np.int64)
        self.indices = keep  # positions in the dataset file (e.g. to look up trained p_z)
        self.names = [str(d["names"][i]) for i in keep]
        frames = np.concatenate([np.arange(d["starts"][i], d["starts"][i] + d["lengths"][i]) for i in keep])
        lengths = d["lengths"][keep]
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
        self.fps = float(d["fps"])
        self.dt = 1.0 / self.fps
        self.num_motions = len(keep)
        self.starts = t(starts, torch.long)
        self.lengths = t(lengths, torch.long)
        self.pos = t(d["pos"][frames])  # (F, B, 3)
        self.rot = quat_to_mat(t(d["quat"][frames]))  # (F, B, 3, 3)
        self.contact = t(d["contact"][frames], torch.bool)
        self.scale = t(d["scale"][keep])
        self.x_nom = t(d["x_nom"][keep])  # (M, B, 3)
        self.z_nom = t(d["z_nom"][keep])
        self.R_nom = t(d["R_nom"])  # (B, 3, 3)
        # finite-difference source velocities (Eqs. 12, 13 need v_m and omega_m)
        nxt = torch.arange(len(frames), device=device) + 1
        last = self.starts + self.lengths - 1
        is_last = torch.zeros(len(frames), dtype=torch.bool, device=device)
        is_last[last] = True
        nxt = torch.where(is_last, nxt - 2, nxt).clamp_min(0)
        sign = torch.where(is_last, -1.0, 1.0).view(-1, 1, 1)
        self.lin_vel = sign * (self.pos[nxt] - self.pos) * self.fps
        dR = self.rot.transpose(-1, -2) @ self.rot[nxt]
        w_local = _so3_log_small(dR) * self.fps
        self.ang_vel = sign * (self.rot @ w_local.unsqueeze(-1)).squeeze(-1)  # world frame
        self.motion_of_frame = torch.repeat_interleave(torch.arange(self.num_motions, device=device), self.lengths)

    def frame_index(self, motion: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = torch.minimum(t, self.lengths[motion] - 1)
        return self.starts[motion] + t


def _so3_log_small(R: torch.Tensor) -> torch.Tensor:
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1) * 0.5).clamp(-1 + 1e-7, 1 - 1e-7)
    th = torch.acos(cos)
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    k = torch.where(th < 1e-4, 0.5 + th**2 / 12, th / (2 * torch.sin(th)))
    return v * k.unsqueeze(-1)


class RetargetParams(torch.nn.Module):
    """Upper-level decision variables p: per-body p_pos, p_ori and per-motion p_z (Eqs. 10-14)."""

    def __init__(self, num_bodies: int, num_motions: int, d_pos: float, d_ori: float, d_z: float):
        super().__init__()
        self.p_pos = torch.nn.Parameter(torch.zeros(num_bodies, 3))
        self.p_ori = torch.nn.Parameter(torch.zeros(num_bodies, 3))
        self.p_z = torch.nn.Parameter(torch.zeros(num_motions))
        self.d_pos, self.d_ori, self.d_z = d_pos, d_ori, d_z

    @torch.no_grad()
    def project(self):
        """Euclidean projection P_P onto the convex set of Eq. 14."""
        for p, r in ((self.p_pos, self.d_pos), (self.p_ori, self.d_ori)):
            n = p.norm(dim=-1, keepdim=True)
            p.mul_(torch.clamp(r / n.clamp_min(1e-12), max=1.0))
        self.p_z.clamp_(-self.d_z, self.d_z)


def map_reference(lib: MotionLib, params: RetargetParams, motion: torch.Tensor, frame: torch.Tensor):
    """Eqs. 10-13: parameterized reference g_t for a batch of (motion, global frame index).

    Differentiable w.r.t. ``params``. Returns body positions, rotations, lin. and ang. velocities,
    each with shape (N, B, ...).
    """
    Rm, xm = lib.rot[frame], lib.pos[frame]
    vm, wm = lib.lin_vel[frame], lib.ang_vel[frame]
    s = lib.scale[motion].view(-1, 1, 1)
    x_nom = lib.x_nom[motion]
    offset = (lib.R_nom.unsqueeze(0) @ params.p_pos.view(1, -1, 3, 1)).squeeze(-1) + x_nom  # (N, B, 3)
    world_off = (Rm @ offset.unsqueeze(-1)).squeeze(-1)
    z = (lib.z_nom[motion] + params.p_z[motion]).view(-1, 1)
    x = world_off + s * xm
    x = torch.cat([x[..., :2], x[..., 2:] + z.unsqueeze(-1)], -1)
    R = Rm @ lib.R_nom.unsqueeze(0) @ exp_map(params.p_ori).unsqueeze(0)
    v = torch.cross(wm, world_off, dim=-1) + s * vm
    return x, R, v, wm
