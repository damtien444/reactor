"""Train ReActor: PPO tracking policy (lower level) + retargeting parameters (upper level)."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from reactor import DATA, ROOT
from reactor.bilevel import UpperCfg, UpperLevel
from reactor.env import EnvCfg, ReActorEnv
from reactor.motion import MotionLib, RetargetParams
from reactor.ppo import PPO, ActorCritic, PPOCfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA / "processed" / "amass_g1.npz")
    ap.add_argument("--run", type=str, default="reactor")
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--no-bilevel", action="store_true", help="RL-only ablation (Fig. 6)")
    ap.add_argument("--warmstart", type=int, default=1000)
    ap.add_argument("--upper-lr", type=float, default=1e-4)
    ap.add_argument("--force-weight", type=float, default=1e-2, help="external wrench penalty (Fig. 8)")
    ap.add_argument("--no-self-collision", action="store_true")
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    out = ROOT / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out)

    lib = MotionLib(args.data, dev)
    ucfg = UpperCfg(lr=args.upper_lr, warmstart=args.warmstart if not args.no_bilevel else 10**12)
    params = RetargetParams(lib.x_nom.shape[1], lib.num_motions, ucfg.d_pos, ucfg.d_ori, ucfg.d_z).to(dev)
    ecfg = EnvCfg(num_envs=args.num_envs, self_collision=not args.no_self_collision,
                  w_force=args.force_weight, w_torque_root=args.force_weight)
    env = ReActorEnv(ecfg, lib, params, dev)
    pcfg = PPOCfg()
    ac = ActorCritic(env.num_obs, env.num_actions, pcfg).to(dev)
    ppo = PPO(ac, pcfg, env.N, env.num_obs, env.num_actions, dev)
    upper = UpperLevel(ucfg, lib, params)
    start_it = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        ac.load_state_dict(ck["ac"])
        params.load_state_dict(ck["params"])
        ppo.opt.load_state_dict(ck["opt"])
        env.fail_count, env.trial_count = ck["fail_count"], ck["trial_count"]
        start_it = ck["iteration"] + 1
    print(f"{lib.num_motions} motions, obs={env.num_obs}, act={env.num_actions}, envs={env.N}")

    obs = env._observe()
    ep_ret = torch.zeros(env.N, device=dev)
    ep_len = torch.zeros(env.N, device=dev)
    done_ret, done_len, done_fell = [], [], []
    t0 = time.time()
    for it in range(start_it, args.iterations):
        tsum = {}
        for _ in range(pcfg.num_steps):
            a = ppo.act(obs)
            obs, r, term, tout, info = env.step(a)
            with torch.no_grad():
                v_to = ac.value(obs) if tout.any() else torch.zeros_like(r)
            ppo.record(r, term, tout, v_to)
            upper.add(info["motion"], info["frame"], info["body_pos"], info["body_rot"], info["valid_upper"])
            for k, v in info["terms"].items():
                tsum[k] = tsum.get(k, 0.0) + v.mean()
            ep_ret += r
            ep_len += 1
            d = term | tout
            if d.any():
                done_ret.append(ep_ret[d].mean().item())
                done_len.append(ep_len[d].mean().item())
                done_fell.append(info["fell"][d].float().mean().item())
                ep_ret[d] = 0
                ep_len[d] = 0
        stats = ppo.update(obs)
        ustats = upper.step(it)

        if it % 10 == 0:
            fps = (it - start_it + 1) * pcfg.num_steps * env.N / (time.time() - t0)
            track = sum(tsum[k] for k in ("root_xy", "root_z", "root_ori", "root_lin", "root_ang",
                                           "rbs_pos", "rbs_ori", "survival")).item() / pcfg.num_steps
            for k, v in stats.items():
                writer.add_scalar(f"ppo/{k}", v, it)
            for k, v in ustats.items():
                writer.add_scalar(f"upper/{k}", v, it)
            for k, v in tsum.items():
                writer.add_scalar(f"reward/{k}", v.item() / pcfg.num_steps, it)
            writer.add_scalar("reward/tracking", track, it)
            if done_ret:
                writer.add_scalar("episode/return", sum(done_ret) / len(done_ret), it)
                writer.add_scalar("episode/length", sum(done_len) / len(done_len), it)
                writer.add_scalar("episode/fall_rate", sum(done_fell) / len(done_fell), it)
            writer.add_scalar("perf/fps", fps, it)
            writer.add_scalar("params/p_pos_norm", params.p_pos.norm(dim=-1).mean().item(), it)
            writer.add_scalar("params/p_ori_norm", params.p_ori.norm(dim=-1).mean().item(), it)
            writer.add_scalar("params/p_z_abs", params.p_z.abs().mean().item(), it)
            print(
                f"it {it:5d} | fps {fps:8.0f} | r_track {track:6.2f} | L {ustats.get('upper_loss', float('nan')):.4f}"
                f" | dp {ustats.get('param_update', 0):.2e} | len {sum(done_len) / max(1, len(done_len)):5.1f}"
                f" | fall {sum(done_fell) / max(1, len(done_fell)):.3f} | kl {stats['kl']:.4f} lr {stats['lr']:.1e}"
                f" | std {ac.log_std.exp().mean().item():.3f}",
                flush=True,
            )
            done_ret, done_len, done_fell = [], [], []
        if it % 500 == 0 or it == args.iterations - 1:
            torch.save(
                {"ac": ac.state_dict(), "params": params.state_dict(), "opt": ppo.opt.state_dict(),
                 "fail_count": env.fail_count, "trial_count": env.trial_count, "iteration": it,
                 "env_cfg": asdict(ecfg), "upper_cfg": asdict(ucfg), "data": str(args.data),
                 "num_obs": env.num_obs, "num_actions": env.num_actions},
                out / "model.pt",
            )


if __name__ == "__main__":
    main()
