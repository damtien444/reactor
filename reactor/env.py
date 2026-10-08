"""Lower-level RL environment (Sec. 6) on MuJoCo Warp.

One control step (50 Hz) = ``decimation`` physics sub-steps, captured as a CUDA graph.
All per-env bookkeeping lives in torch tensors that alias the Warp arrays (zero copy).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from reactor.motion import MotionLib, RetargetParams, map_reference
from reactor.robot import RobotInfo, build_model
from reactor.rotations import geodesic_sq, mat_to_6d, quat_to_mat


@dataclass
class EnvCfg:
    num_envs: int = 4096
    sim_dt: float = 0.005
    decimation: int = 4  # 50 Hz control
    self_collision: bool = True
    init_steps: int = 50  # duration of the retargeting phase psi: 0 -> 1 (1 s)
    max_episode_steps: int = 500
    init_joint_std: float = 0.1
    deadband: float = 0.1  # Eq. 19
    force_scale: float = 100.0  # N per unit wrench action
    torque_scale: float = 20.0  # Nm per unit wrench action
    action_clip: float = 10.0
    goal_offsets: tuple = (1, 5)
    term_pos: float = 0.5  # early termination on torso tracking (adaptive sampling failures)
    term_ori: float = float(np.deg2rad(45.0))
    nconmax: int = 48
    njmax: int = 320
    # reward weights, Table 1
    w_root_xy: float = 2.0
    w_root_z: float = 10.0
    w_root_ori: float = 2.0
    w_root_lin: float = 0.5
    w_root_ang: float = 0.5
    w_rbs_pos: float = 2.0
    w_rbs_ori: float = 2.0
    w_survival: float = 20.0
    w_torque: float = 1e-4
    w_qacc: float = 1e-6
    w_action_rate: float = 1e-2
    w_action_acc: float = 1e-2
    w_force: float = 1e-2
    w_torque_root: float = 1e-2
    sampling_floor: float = 0.05  # probability mass kept uniform in adaptive sampling


class ReActorEnv:
    def __init__(self, cfg: EnvCfg, lib: MotionLib, params: RetargetParams, device: torch.device):
        self.cfg, self.lib, self.params, self.device = cfg, lib, params, device
        self.mj_model = build_model(cfg.sim_dt, cfg.self_collision)
        self.robot = RobotInfo(self.mj_model)
        N, m = cfg.num_envs, self.mj_model
        self.N = N
        self.nj = self.robot.num_joints
        self.num_actions = self.nj + 6
        self.dt = cfg.sim_dt * cfg.decimation

        mjd = mujoco.MjData(m)
        mjd.qpos[:] = m.qpos0
        mjd.qpos[7:] = self.robot.default_q
        mjd.qpos[2] = 0.8
        mujoco.mj_forward(m, mjd)
        wp.init()
        # Warp cannot capture graphs on torch's legacy default stream, so run torch and the
        # simulator on one shared non-default stream.
        torch.cuda.set_stream(torch.cuda.Stream(device))
        self.wp_stream = wp.stream_from_torch(torch.cuda.current_stream(device))
        with wp.ScopedStream(self.wp_stream):
            self.wm = mjw.put_model(m)
            self.wd = mjw.put_data(m, mjd, nworld=N, nconmax=cfg.nconmax, njmax=cfg.njmax)
            self._capture()

        tt = wp.to_torch
        self.qpos, self.qvel, self.ctrl = tt(self.wd.qpos), tt(self.wd.qvel), tt(self.wd.ctrl)
        self.xpos, self.xmat = tt(self.wd.xpos), tt(self.wd.xmat)
        self.xfrc = tt(self.wd.xfrc_applied)
        self.act_force = tt(self.wd.actuator_force)
        self.qacc_ws = tt(self.wd.qacc_warmstart)
        self.geom_xpos, self.geom_xmat = tt(self.wd.geom_xpos), tt(self.wd.geom_xmat)
        self.time = tt(self.wd.time)

        r = self.robot
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
        self.default_q = t(r.default_q)
        self.action_scale = t(r.action_scale)
        self.body_ids = t(r.body_ids, torch.long)
        self.root = r.root_body
        self.torso_pair = [b for b, _ in __import__("reactor.robot", fromlist=["x"]).CORRESPONDENCES].index("torso_link")
        col = [g for g in range(m.ngeom) if m.geom_bodyid[g] > 0 and m.geom_contype[g]]
        self.col_geoms = t(col, torch.long)
        self.col_radius = t(m.geom_size[col, 0])
        self.col_half = t(np.where(m.geom_type[col] == mujoco.mjtGeom.mjGEOM_CAPSULE, m.geom_size[col, 1], 0.0))
        self.qpos0 = t(m.qpos0)

        # per-env state
        self.motion = torch.zeros(N, dtype=torch.long, device=device)
        self.start = torch.zeros(N, dtype=torch.long, device=device)
        self.k = torch.zeros(N, dtype=torch.long, device=device)
        self.actions = torch.zeros(N, self.num_actions, device=device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.prev_prev_actions = torch.zeros_like(self.actions)
        self.prev_qvel = torch.zeros(N, self.nj, device=device)
        self.wrench = torch.zeros(N, 6, device=device)
        # adaptive sampling statistics
        M = lib.num_motions
        self.fail_count = torch.ones(M, device=device)
        self.trial_count = torch.full((M,), 2.0, device=device)

        self.num_obs = self._observe().shape[1]
        self.reset(torch.arange(N, device=device))

    # ------------------------------------------------------------------ simulation
    def _capture(self):
        def physics():
            for _ in range(self.cfg.decimation):
                mjw.step(self.wm, self.wd)

        physics()  # compile kernels before capture
        mjw.kinematics(self.wm, self.wd)
        with wp.ScopedCapture() as cap:
            physics()
        self.g_physics = cap.graph
        with wp.ScopedCapture() as cap:
            mjw.kinematics(self.wm, self.wd)
            mjw.com_pos(self.wm, self.wd)
        self.g_kin = cap.graph

    def _launch(self, graph):
        with wp.ScopedStream(self.wp_stream):
            wp.capture_launch(graph)

    # ------------------------------------------------------------------ reference
    def ref_frame(self, k: torch.Tensor) -> torch.Tensor:
        """Global frame index for control counter k (reference paused during the init phase)."""
        t = self.start + (k - self.cfg.init_steps).clamp_min(0)
        return self.lib.frame_index(self.motion, t)

    def psi(self) -> torch.Tensor:
        return (self.k.float() / self.cfg.init_steps).clamp(max=1.0)

    def reference(self, k: torch.Tensor, grad: bool = False):
        with torch.set_grad_enabled(grad):
            return map_reference(self.lib, self.params, self.motion, self.ref_frame(k))

    # ------------------------------------------------------------------ robot state
    def body_state(self):
        return self.xpos[:, self.body_ids], self.xmat[:, self.body_ids]

    def _observe(self) -> torch.Tensor:
        R_root = self.xmat[:, self.root]  # (N,3,3)
        p_root = self.qpos[:, :3]
        RT = R_root.transpose(-1, -2)
        lin = (RT @ self.qvel[:, :3].unsqueeze(-1)).squeeze(-1)
        ang = self.qvel[:, 3:6]
        grav = RT[:, :, 2] * -1.0
        q = self.qpos[:, 7:] - self.default_q
        qd = self.qvel[:, 6:]
        obs = [p_root[:, 2:3], grav, lin, ang, q, qd * 0.1, self.prev_actions, self.prev_prev_actions,
               self.psi().unsqueeze(-1)]
        for off in self.cfg.goal_offsets:
            x, R, v, w = self.reference(self.k + off)
            rel = (RT.unsqueeze(1) @ (x - p_root.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
            rrel = RT.unsqueeze(1) @ R
            vroot = (RT @ v[:, 0].unsqueeze(-1)).squeeze(-1)
            wroot = (RT @ w[:, 0].unsqueeze(-1)).squeeze(-1)
            obs += [rel.flatten(1), mat_to_6d(rrel).flatten(1), vroot, wroot]
        return torch.cat(obs, -1)

    # ------------------------------------------------------------------ reset
    def _sample_motions(self, n: int) -> torch.Tensor:
        rate = self.fail_count / self.trial_count
        p = rate / rate.sum()
        p = (1 - self.cfg.sampling_floor) * p + self.cfg.sampling_floor / p.numel()
        return torch.multinomial(p, n, replacement=True)

    def reset(self, ids: torch.Tensor, motions: torch.Tensor | None = None, starts: torch.Tensor | None = None):
        if ids.numel() == 0:
            return
        n = ids.numel()
        mot = self._sample_motions(n) if motions is None else motions
        L = self.lib.lengths[mot]
        if starts is None:
            starts = (torch.rand(n, device=self.device) * (L - 1).clamp_min(1).float()).long()
        start = starts
        self.motion[ids], self.start[ids], self.k[ids] = mot, start, 0
        self.trial_count[mot] += 1  # counted at start, failures counted on early termination

        # root state from the source root mapped through the current parameters (pair 0)
        with torch.no_grad():
            x, R, _, _ = map_reference(self.lib, self.params, mot, self.lib.frame_index(mot, start))
        from reactor.rotations import mat_to_quat

        qpos = self.qpos0.expand(n, -1).clone()
        qpos[:, :3] = x[:, 0]
        qpos[:, 3:7] = mat_to_quat(R[:, 0])
        q = self.default_q + self.cfg.init_joint_std * torch.randn(n, self.nj, device=self.device)
        lo, hi = torch.as_tensor(self.robot.joint_range.T, dtype=torch.float32, device=self.device)
        qpos[:, 7:] = torch.max(torch.min(q, hi), lo)
        self.qpos[ids] = qpos
        self.qvel[ids] = 0.0
        self.qacc_ws[ids] = 0.0
        self.ctrl[ids] = qpos[:, 7:]
        self.xfrc[ids] = 0.0
        for buf in (self.actions, self.prev_actions, self.prev_prev_actions, self.wrench):
            buf[ids] = 0.0
        self.prev_qvel[ids] = 0.0
        # lift so no collision geom starts below the ground
        self._launch(self.g_kin)
        gz = self.geom_xpos[ids][:, self.col_geoms, 2]
        axz = self.geom_xmat[ids][:, self.col_geoms, 2, 2].abs()
        lowest = (gz - self.col_radius - self.col_half * axz).min(-1).values
        self.qpos[ids, 2] += (0.005 - lowest).clamp_min(0.0)
        self._launch(self.g_kin)

    # ------------------------------------------------------------------ step
    def step(self, actions: torch.Tensor):
        c = self.cfg
        a = actions.clamp(-c.action_clip, c.action_clip)
        self.prev_prev_actions.copy_(self.prev_actions)
        self.prev_actions.copy_(self.actions)
        self.actions.copy_(a)
        self.ctrl.copy_(self.default_q + self.action_scale * a[:, : self.nj])
        w = a[:, self.nj :]
        w = torch.sign(w) * (w.abs() - c.deadband).clamp_min(0.0)  # Eq. 19
        self.wrench = torch.cat([w[:, :3] * c.force_scale, w[:, 3:] * c.torque_scale], -1)
        self.xfrc[:, self.root] = self.wrench
        self.prev_qvel.copy_(self.qvel[:, 6:])

        self._launch(self.g_physics)
        self.k += 1

        reward, terms, err = self._reward()
        # no early termination while the robot is still moving to the start pose (psi < 1)
        fell = ((err["root_pos"] > c.term_pos) | (err["torso_ori"] > c.term_ori)) & (self.psi() >= 1.0)
        bad = ~torch.isfinite(self.qpos).all(-1) | ~torch.isfinite(self.qvel).all(-1)
        t_motion = self.start + (self.k - c.init_steps).clamp_min(0)
        timeout = (t_motion >= self.lib.lengths[self.motion] - 1) | (self.k >= c.max_episode_steps)
        terminated = fell | bad
        done = terminated | timeout
        reward = torch.where(bad, torch.zeros_like(reward), reward)

        info = {"terms": terms, "psi": self.psi(), "motion": self.motion.clone(),
                "frame": self.ref_frame(self.k), "fell": fell}
        # upper-level batch: simulated body frames of envs that finished the retargeting phase
        bp, bR = self.body_state()
        info["body_pos"], info["body_rot"] = bp.clone(), bR.clone()
        info["valid_upper"] = (self.psi() >= 1.0) & ~bad

        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel():
            self.fail_count.index_add_(0, self.motion[ids], fell[ids].float())
            if bad.any():
                bad_ids = bad.nonzero(as_tuple=False).squeeze(-1)
                self.qpos[bad_ids] = self.qpos0
                self.qvel[bad_ids] = 0.0
            self.reset(ids)
        obs = self._observe()
        return obs, reward, terminated, timeout, info

    def _reward(self):
        c = self.cfg
        psi = self.psi()
        x, R, v, w = self.reference(self.k)
        bp, bR = self.body_state()
        root_lin = self.qvel[:, :3]
        root_ang = (self.xmat[:, self.root] @ self.qvel[:, 3:6].unsqueeze(-1)).squeeze(-1)

        e_xy = ((x[:, 0, :2] - bp[:, 0, :2]) ** 2).sum(-1)
        e_z = (x[:, 0, 2] - bp[:, 0, 2]) ** 2
        e_ori = geodesic_sq(bR[:, 0], R[:, 0])
        e_lin = ((v[:, 0] - root_lin) ** 2).sum(-1)
        e_ang = ((w[:, 0] - root_ang) ** 2).sum(-1)
        e_pos_b = ((x - bp) ** 2).sum(-1)  # (N, B)
        e_ori_b = geodesic_sq(bR, R)
        e_rbs_pos = e_pos_b[:, 1:].sum(-1)
        e_rbs_ori = e_ori_b[:, 1:].sum(-1)

        qacc = (self.qvel[:, 6:] - self.prev_qvel) / self.dt
        tau = self.act_force
        a, a1, a2 = self.actions[:, : self.nj], self.prev_actions[:, : self.nj], self.prev_prev_actions[:, : self.nj]
        terms = {
            "root_xy": -c.w_root_xy * e_xy,
            "root_z": -c.w_root_z * e_z,
            "root_ori": -c.w_root_ori * e_ori,
            "root_lin": -c.w_root_lin * e_lin,
            "root_ang": -c.w_root_ang * e_ang,
            "rbs_pos": -c.w_rbs_pos * psi * e_rbs_pos,
            "rbs_ori": -c.w_rbs_ori * psi * e_rbs_ori,
            "survival": torch.full_like(psi, c.w_survival),
            "torque": -c.w_torque * (tau**2).sum(-1),
            "qacc": -c.w_qacc * (qacc**2).sum(-1),
            "action_rate": -c.w_action_rate * ((a - a1) ** 2).sum(-1),
            "action_acc": -c.w_action_acc * ((a - 2 * a1 + a2) ** 2).sum(-1),
            "root_force": -c.w_force * psi * self.wrench[:, :3].abs().sum(-1),
            "root_torque": -c.w_torque_root * psi * self.wrench[:, 3:].abs().sum(-1),
        }
        reward = sum(terms.values())
        tp = self.torso_pair
        err = {
            "root_pos": (x[:, 0] - bp[:, 0]).norm(dim=-1),
            "torso_ori": geodesic_sq(bR[:, tp], R[:, tp]).sqrt(),
        }
        return reward, terms, err
