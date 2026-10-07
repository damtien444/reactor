"""Upper-level optimization (Sec. 4): single-loop TTSA with the simplified gradient of Eq. 6-7."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from reactor.motion import MotionLib, RetargetParams, map_reference
from reactor.rotations import log_map


@dataclass
class UpperCfg:
    lr: float = 1e-4  # eta in Eq. 3
    alpha: float = 0.0  # sensitivity of optimal trajectories, Eq. 5; (1 - alpha) folds into eta
    warmstart: int = 1000  # RL-only iterations before parameters are updated (Fig. 6)
    every: int = 1
    w_x: float = 10.0
    w_R: float = 1.0
    w_v: float = 0.0
    w_w: float = 0.0
    d_pos: float = 0.5
    d_ori: float = 0.5
    d_z: float = 0.5
    max_samples: int = 65536


class UpperLevel:
    def __init__(self, cfg: UpperCfg, lib: MotionLib, params: RetargetParams):
        self.cfg, self.lib, self.params = cfg, lib, params
        self.reset_batch()

    def reset_batch(self):
        self.buf = {"motion": [], "frame": [], "pos": [], "rot": []}

    @torch.no_grad()
    def add(self, motion, frame, body_pos, body_rot, valid):
        """Collect (s_t, g_t) pairs; only segments with psi = 1 are used (Sec. 6)."""
        idx = valid.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return
        self.buf["motion"].append(motion[idx])
        self.buf["frame"].append(frame[idx])
        self.buf["pos"].append(body_pos[idx])
        self.buf["rot"].append(body_rot[idx])

    def loss(self, motion, frame, pos, rot):
        c = self.cfg
        x, R, _, _ = map_reference(self.lib, self.params, motion, frame)
        l_x = ((x - pos) ** 2).sum(-1)
        l_R = (log_map(rot.transpose(-1, -2) @ R) ** 2).sum(-1)  # Eq. 16
        return (c.w_x * l_x + c.w_R * l_R).sum(-1)  # Eq. 17 (w_v = w_w = 0, Tab. 2)

    def step(self, iteration: int) -> dict:
        c = self.cfg
        if not self.buf["motion"]:
            return {}
        batch = {k: torch.cat(v) for k, v in self.buf.items()}
        self.reset_batch()
        n = batch["motion"].shape[0]
        if n > c.max_samples:
            sel = torch.randperm(n, device=batch["motion"].device)[: c.max_samples]
            batch = {k: v[sel] for k, v in batch.items()}
        p = self.params
        p.zero_grad(set_to_none=True)
        loss = self.loss(batch["motion"], batch["frame"], batch["pos"], batch["rot"]).mean()  # Eq. 7
        out = {"upper_loss": loss.item(), "upper_samples": n}
        if iteration < c.warmstart or iteration % c.every:
            out["param_update"] = 0.0
            return out
        loss.backward()
        with torch.no_grad():
            old = [t.detach().clone() for t in p.parameters()]
            # p_z is per motion: normalize its gradient by the motion's share of the batch so
            # each motion's offset is updated by the mean gradient over its own samples.
            counts = torch.bincount(batch["motion"], minlength=self.lib.num_motions).float()
            # p_z is shared by all bodies of a motion: also average over bodies so its step size
            # matches the per-body p_pos steps.
            p.p_z.grad *= batch["motion"].shape[0] / counts.clamp_min(1.0) / self.lib.x_nom.shape[1]
            for t in p.parameters():
                t -= c.lr * (1.0 - c.alpha) * t.grad  # Eq. 3 with Eq. 6
            p.project()
            upd = torch.sqrt(sum(((t - o) ** 2).sum() for t, o in zip(p.parameters(), old)))
        out["param_update"] = upd.item()
        return out
