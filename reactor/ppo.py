"""PPO with adaptive learning rate (Rudin et al. 2022), Table 2 hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PPOCfg:
    num_steps: int = 24
    num_minibatches: int = 4
    num_epochs: int = 5
    clip: float = 0.2
    entropy_coef: float = 0.0025
    value_coef: float = 1.0
    gamma: float = 0.97
    lam: float = 0.95
    desired_kl: float = 0.009
    max_grad_norm: float = 1.0
    lr: float = 1e-3
    init_std: float = 0.5
    hidden: tuple = (512, 512, 512)


def mlp(inp, out, hidden):
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    def __init__(self, dim, eps=1e-2, clip=5.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.eps, self.clip = eps, clip

    @torch.no_grad()
    def update(self, x):
        m, v, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        tot = self.count + n
        delta = m - self.mean
        self.mean += delta * n / tot
        self.var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + self.eps)).clamp(-self.clip, self.clip)


class ActorCritic(nn.Module):
    def __init__(self, num_obs, num_actions, cfg: PPOCfg):
        super().__init__()
        self.norm = RunningNorm(num_obs)
        self.actor = mlp(num_obs, num_actions, cfg.hidden)
        self.critic = mlp(num_obs, 1, cfg.hidden)
        self.log_std = nn.Parameter(torch.full((num_actions,), float(torch.log(torch.tensor(cfg.init_std)))))

    def dist(self, obs):
        mu = self.actor(self.norm(obs))
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, obs):
        return self.critic(self.norm(obs)).squeeze(-1)

    @torch.no_grad()
    def act_inference(self, obs):
        return self.actor(self.norm(obs))


class PPO:
    def __init__(self, ac: ActorCritic, cfg: PPOCfg, num_envs: int, num_obs: int, num_actions: int, device):
        self.ac, self.cfg = ac, cfg
        self.opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr)
        self.lr = cfg.lr
        T, N = cfg.num_steps, num_envs
        z = lambda *s: torch.zeros(*s, device=device)
        self.obs = z(T, N, num_obs)
        self.actions = z(T, N, num_actions)
        self.logp = z(T, N)
        self.mu = z(T, N, num_actions)
        self.sigma = z(T, N, num_actions)
        self.values = z(T, N)
        self.rewards = z(T, N)
        self.dones = z(T, N)
        self.step = 0

    @torch.no_grad()
    def act(self, obs):
        d = self.ac.dist(obs)
        a = d.sample()
        i = self.step
        self.obs[i], self.actions[i] = obs, a
        self.logp[i] = d.log_prob(a).sum(-1)
        self.mu[i], self.sigma[i] = d.mean, d.stddev
        self.values[i] = self.ac.value(obs)
        return a

    @torch.no_grad()
    def record(self, reward, terminated, timeout, next_obs_value_on_timeout):
        i = self.step
        # bootstrap on time-outs
        self.rewards[i] = reward + self.cfg.gamma * next_obs_value_on_timeout * timeout.float()
        self.dones[i] = (terminated | timeout).float()
        self.step += 1

    @torch.no_grad()
    def _returns(self, last_value):
        c = self.cfg
        adv = torch.zeros_like(self.rewards)
        gae = 0
        for t in reversed(range(c.num_steps)):
            nv = last_value if t == c.num_steps - 1 else self.values[t + 1]
            nd = 1.0 - self.dones[t]
            delta = self.rewards[t] + c.gamma * nv * nd - self.values[t]
            gae = delta + c.gamma * c.lam * nd * gae
            adv[t] = gae
        return adv + self.values, adv

    def update(self, last_obs):
        c = self.cfg
        self.ac.norm.update(self.obs.flatten(0, 1))
        with torch.no_grad():
            returns, adv = self._returns(self.ac.value(last_obs))
        B = self.obs.shape[0] * self.obs.shape[1]
        flat = lambda x: x.reshape(B, *x.shape[2:])
        obs, act, logp0 = flat(self.obs), flat(self.actions), flat(self.logp)
        mu0, sig0, ret, advf, v0 = flat(self.mu), flat(self.sigma), flat(returns), flat(adv), flat(self.values)
        mb = B // c.num_minibatches
        stats = {"value_loss": 0.0, "surrogate": 0.0, "kl": 0.0, "entropy": 0.0}
        n = 0
        for _ in range(c.num_epochs):
            perm = torch.randperm(B, device=obs.device)
            for k in range(c.num_minibatches):
                idx = perm[k * mb : (k + 1) * mb]
                d = self.ac.dist(obs[idx])
                logp = d.log_prob(act[idx]).sum(-1)
                ent = d.entropy().sum(-1).mean()
                with torch.no_grad():
                    kl = (
                        torch.log(d.stddev / sig0[idx])
                        + (sig0[idx] ** 2 + (mu0[idx] - d.mean) ** 2) / (2 * d.stddev**2)
                        - 0.5
                    ).sum(-1).mean()
                    if kl > 2 * c.desired_kl:
                        self.lr = max(1e-5, self.lr / 1.5)
                    elif 0 < kl < c.desired_kl / 2:
                        self.lr = min(1e-2, self.lr * 1.5)
                    for g in self.opt.param_groups:
                        g["lr"] = self.lr
                a = advf[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = torch.exp(logp - logp0[idx])
                surr = -torch.min(ratio * a, ratio.clamp(1 - c.clip, 1 + c.clip) * a).mean()
                v = self.ac.value(obs[idx])
                v_clip = v0[idx] + (v - v0[idx]).clamp(-c.clip, c.clip)
                vl = torch.max((v - ret[idx]) ** 2, (v_clip - ret[idx]) ** 2).mean()
                loss = surr + c.value_coef * vl - c.entropy_coef * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), c.max_grad_norm)
                self.opt.step()
                stats["value_loss"] += vl.item()
                stats["surrogate"] += surr.item()
                stats["kl"] += kl.item()
                stats["entropy"] += ent.item()
                n += 1
        self.step = 0
        return {k: v / n for k, v in stats.items()} | {"lr": self.lr}
