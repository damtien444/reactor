"""Retarget every motion with a trained ReActor policy and compute the Table 3 metrics.

Each motion is played once from frame 0 (after the 1 s retargeting phase) with the
deterministic policy. The simulated trajectory *is* the retargeted motion. We report
ground penetration, self-penetration, foot sliding and foot floating against the
estimated source contacts, plus the retargeting success rate (no root/torso failure) and
the applied external wrench (Fig. 8). The retargeted joint trajectories are saved for
downstream tracking.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch

from reactor import ROOT
from reactor.env import EnvCfg, ReActorEnv
from reactor.motion import MotionLib, RetargetParams
from reactor.ppo import ActorCritic, PPOCfg


def foot_sole_points(env: ReActorEnv):
    m = env.mj_model
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in env.robot.foot_ids and m.geom_contype[g]]
    side = torch.tensor([list(env.robot.foot_ids).index(m.geom_bodyid[g]) for g in geoms], device=env.device)
    return torch.tensor(geoms, device=env.device), side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--max-motions", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda")
    out = ROOT / "runs" / args.run
    ck = torch.load(out / "model.pt", map_location=dev, weights_only=False)
    lib = MotionLib(Path(ck["data"]), dev)
    uc = ck["upper_cfg"]
    params = RetargetParams(lib.x_nom.shape[1], lib.num_motions, uc["d_pos"], uc["d_ori"], uc["d_z"]).to(dev)
    params.load_state_dict(ck["params"])
    ecfg = EnvCfg(**(ck["env_cfg"] | {"num_envs": args.num_envs, "max_episode_steps": 10**9}))
    env = ReActorEnv(ecfg, lib, params, dev)
    ac = ActorCritic(ck["num_obs"], ck["num_actions"], PPOCfg()).to(dev)
    ac.load_state_dict(ck["ac"])
    ac.eval()

    m = env.mj_model
    sole_geoms, sole_side = foot_sole_points(env)
    col = env.col_geoms
    body_of_col = torch.as_tensor(m.geom_bodyid[col.cpu().numpy()], device=dev)
    # kinematic chains to ignore for self-penetration: parent/child body pairs
    parent = torch.as_tensor(m.body_parentid, device=dev)
    nb = m.nbody
    adjacent = torch.zeros(nb, nb, dtype=torch.bool, device=dev)
    for b in range(1, nb):
        adjacent[b, b] = True
        adjacent[b, parent[b]] = adjacent[parent[b], b] = True
    ga, gb = torch.triu_indices(len(col), len(col), 1, device=dev)
    keep = ~adjacent[body_of_col[ga], body_of_col[gb]]
    ga, gb = ga[keep], gb[keep]

    M = lib.num_motions if not args.max_motions else min(args.max_motions, lib.num_motions)
    results = {k: np.zeros(M) for k in
               ("gp_time", "gp_depth", "sp_time", "sp_depth", "slide", "float", "success", "force", "torque")}
    joints_out = {}
    N = env.N
    for b0 in range(0, M, N):
        ids = torch.arange(b0, min(M, b0 + N), device=dev)
        n = ids.numel()
        motions = torch.cat([ids, ids[:1].expand(N - n)])
        env.reset(torch.arange(N, device=dev), motions, torch.zeros_like(motions))
        obs = env._observe()
        L = lib.lengths[ids]
        T = int(L.max().item()) + ecfg.init_steps
        acc = {k: torch.zeros(n, device=dev) for k in results}
        cnt = torch.zeros(n, device=dev)
        cnt_contact = torch.zeros(n, device=dev)
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        qs = torch.zeros(n, int(L.max().item()), 3 + 4 + env.nj, device=dev)
        prev_foot = None
        for k in range(T):
            with torch.no_grad():
                a = ac.act_inference(obs)
            obs, _, term, tout, info = env.step(a)
            kk = env.k[:n] if k < T - 1 else None
            t = k + 1 - ecfg.init_steps  # motion frame of the state after this step
            active = alive & (t >= 0) & (t < L)
            fell = info["fell"][:n] & active
            alive &= ~fell
            # env auto-resets on done; we stop counting a motion after its first termination
            if t < 0:
                continue
            frame = lib.starts[ids] + torch.minimum(torch.full_like(L, t), L - 1)
            act_f = active.float()
            gz = env.geom_xpos[:n][:, col, 2]
            axz = env.geom_xmat[:n][:, col, 2, 2].abs()
            low = gz - env.col_radius - env.col_half * axz
            pen = (-low).clamp_min(0).max(-1).values
            acc["gp_time"] += act_f * (pen > 0.01).float()
            acc["gp_depth"] += act_f * pen
            # self-penetration from capsule/sphere segment distances
            p = env.geom_xpos[:n][:, col]
            ax = env.geom_xmat[:n][:, col][..., :, 2]
            sp = _capsule_pen(p[:, ga], ax[:, ga], env.col_half[ga], env.col_radius[ga],
                              p[:, gb], ax[:, gb], env.col_half[gb], env.col_radius[gb])
            spm = sp.max(-1).values
            acc["sp_time"] += act_f * (spm > 0.0).float() * lib.dt
            acc["sp_depth"] = torch.maximum(acc["sp_depth"], act_f * spm)
            # feet: sliding velocity and floating height during source contact
            fpos = env.xpos[:n][:, env.robot.foot_ids]
            sole = low[:, [list(col.cpu().numpy()).index(g) for g in sole_geoms.cpu().numpy()]]
            sole_h = torch.stack([sole[:, sole_side == s].min(-1).values for s in range(2)], -1).clamp_min(0)
            c = lib.contact[frame] & active.unsqueeze(-1)
            if prev_foot is not None:
                v = ((fpos - prev_foot)[..., :2].norm(dim=-1)) / env.dt
                acc["slide"] += (v * c).sum(-1)
            acc["float"] += (sole_h * c).sum(-1)
            cnt_contact += c.float().sum(-1)
            prev_foot = fpos.clone()
            acc["force"] = torch.maximum(acc["force"], act_f * env.wrench[:n, :3].norm(dim=-1))
            acc["torque"] = torch.maximum(acc["torque"], act_f * env.wrench[:n, 3:].norm(dim=-1))
            cnt += act_f
            tt = min(t, qs.shape[1] - 1)
            qs[:, tt] = torch.where(active.unsqueeze(-1), env.qpos[:n], qs[:, tt])
        acc["success"] = alive.float()
        for key in results:
            v = acc[key]
            if key in ("gp_time", "gp_depth"):
                v = v / cnt.clamp_min(1)
            elif key in ("slide", "float"):
                v = v / cnt_contact.clamp_min(1)
            results[key][b0 : b0 + n] = v.cpu().numpy()
        for i in range(n):
            mi = int(ids[i])
            joints_out[lib.names[mi]] = qs[i, : int(L[i])].cpu().numpy()
        print(f"motions {b0 + n}/{M}  success so far {results['success'][: b0 + n].mean():.3f}", flush=True)

    fmt = lambda k, s=1.0: f"{(results[k] * s).mean():.2f} ± {(results[k] * s).std():.2f}"
    lines = [
        f"run: {args.run}   motions: {M}",
        f"Ground-pen.  time {fmt('gp_time')}   depth [cm] {fmt('gp_depth', 100)}",
        f"Self-pen.    time [s] {fmt('sp_time')}   depth [cm] {fmt('sp_depth', 100)}",
        f"Foot slide   [cm/s] {fmt('slide', 100)}",
        f"Foot float   [cm] {fmt('float', 100)}",
        f"Retarget success [%] {100 * results['success'].mean():.2f}",
        f"Max ext. force [N] {fmt('force')}   torque [Nm] {fmt('torque')}",
    ]
    print("\n".join(lines))
    (out / "eval.txt").write_text("\n".join(lines) + "\n")
    np.savez(out / "eval_metrics.npz", names=np.array(lib.names[:M]), **results)
    np.savez_compressed(out / "retargeted_qpos.npz", **{k.replace("/", "|"): v for k, v in joints_out.items()})


def _capsule_pen(p1, a1, h1, r1, p2, a2, h2, r2):
    """Penetration depth between capsules (spheres have h = 0) via segment-segment distance."""
    d1, d2 = a1 * h1.unsqueeze(-1), a2 * h2.unsqueeze(-1)
    s1, s2 = p1 - d1, p2 - d2
    e1, e2 = 2 * d1, 2 * d2
    r = s1 - s2
    a = (e1 * e1).sum(-1)
    e = (e2 * e2).sum(-1)
    f = (e2 * r).sum(-1)
    c = (e1 * r).sum(-1)
    b = (e1 * e2).sum(-1)
    denom = (a * e - b * b).clamp_min(1e-9)
    s = torch.where(a > 1e-9, ((b * f - c * e) / denom).clamp(0, 1), torch.zeros_like(a))
    t = torch.where(e > 1e-9, (b * s + f) / e.clamp_min(1e-9), torch.zeros_like(a))
    t_c = t.clamp(0, 1)
    s = torch.where((t < 0) | (t > 1), torch.where(a > 1e-9, ((b * t_c - c) / a.clamp_min(1e-9)).clamp(0, 1), s), s)
    c1 = s1 + e1 * s.unsqueeze(-1)
    c2 = s2 + e2 * t_c.unsqueeze(-1)
    dist = (c1 - c2).norm(dim=-1)
    return (r1 + r2 - dist).clamp_min(0)


if __name__ == "__main__":
    main()
