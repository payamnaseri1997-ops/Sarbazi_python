# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 01:14:25 2025

@author: elecomp
"""

# ============================
# === rl_sac.py (DEEP SAC) ===
# ============================
"""Deep residual RL policy using Soft Actor-Critic (SAC) in PyTorch.
Provides save()/load() so it can be persisted and reused by name.
"""
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json

class ResidualAgentAPI:  # minimal duck-typing to satisfy main
    def act(self, obs, eval: bool=False) -> float: ...
    def begin_episode(self): ...
    def end_episode(self): ...
    def observe(self, o, a, r, o2, d): ...
    def update(self): ...
    def save(self, name: str, out_dir: str = "agents"): ...

class MLP(nn.Module):
    def __init__(self, inp, out, hidden=(128,128), act=nn.ReLU):
        super().__init__()
        layers = []
        last = inp
        for h in hidden:
            layers += [nn.Linear(last, h), act()]
            last = h
        layers += [nn.Linear(last, out)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, u_max):
        super().__init__()
        self.body = MLP(obs_dim, 2*act_dim)
        self.u_max = float(u_max)
    def forward(self, x):
        mu_logstd = self.body(x)
        mu, logstd = torch.chunk(mu_logstd, 2, dim=-1)
        logstd = torch.clamp(logstd, -5, 2)
        std = torch.exp(logstd)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        a = torch.tanh(z)
        a_scaled = a * self.u_max
        logp = dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        logp = logp.sum(dim=-1, keepdim=True)
        return a_scaled, logp, mu.tanh()*self.u_max

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q = MLP(obs_dim + act_dim, 1)
    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1))

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=200000):
        self.size = size
        self.ptr = 0
        self.len = 0
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros((size, 1), dtype=np.float32)
        self.obs2 = np.zeros((size, obs_dim), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)
    def store(self, o, a, r, o2, d):
        i = self.ptr % self.size
        self.obs[i] = o
        self.act[i] = a
        self.rew[i] = r
        self.obs2[i] = o2
        self.done[i] = d
        self.ptr = (self.ptr + 1) % self.size
        self.len = min(self.len + 1, self.size)
    def sample_batch(self, batch_size):
        idx = np.random.randint(0, self.len, size=batch_size)
        return dict(obs=torch.as_tensor(self.obs[idx]),
                    act=torch.as_tensor(self.act[idx]),
                    rew=torch.as_tensor(self.rew[idx]),
                    obs2=torch.as_tensor(self.obs2[idx]),
                    done=torch.as_tensor(self.done[idx]))

@dataclass
class SACConfig:
    u_rl_max: float = 0.18     # residual torque bound
    gamma: float = 0.997       # discount
    tau: float = 0.005         # target smoothing
    lr: float = 3e-4           # learning rate
    batch_size: int = 256      # update batch size
    start_steps: int = 2000    # random steps before using policy
    updates_per_step: int = 1  # gradient steps per env step
    alpha: float = 0.2         # initial temperature
    autotune_alpha: bool = True

class SACResidualPolicy(ResidualAgentAPI):
    """Soft Actor-Critic residual policy. Actions are tanh-squashed and scaled to ±u_rl_max.
    Observation features are identical to SIMPLE agent: [e, edot, s, omega, 1.0, time_frac] (clipped to ±5).
    """
    def __init__(self, obs_dim: int, act_dim: int, cfg: SACConfig):
        self.device = torch.device('cpu')
        self.cfg = cfg
        self.actor = Actor(obs_dim, act_dim, cfg.u_rl_max).to(self.device)
        self.q1 = Critic(obs_dim, act_dim).to(self.device)
        self.q2 = Critic(obs_dim, act_dim).to(self.device)
        self.q1_t = Critic(obs_dim, act_dim).to(self.device)
        self.q2_t = Critic(obs_dim, act_dim).to(self.device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.pi_optim = optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=cfg.lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=cfg.lr)
        if cfg.autotune_alpha:
            self.log_alpha = torch.tensor(np.log(cfg.alpha), requires_grad=True)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=cfg.lr)
            self.target_entropy = -act_dim
        else:
            self.log_alpha = torch.tensor(np.log(cfg.alpha), requires_grad=False)
            self.alpha_optim = None
            self.target_entropy = None
        self.replay = ReplayBuffer(obs_dim, act_dim)
        self.total_steps = 0

    # ---- API ----
    def _encode_obs(self, obs_dict):
        phi = np.array([obs_dict['e'], obs_dict['edot'], obs_dict['s'], obs_dict['omega'], 1.0, obs_dict['time_frac']], dtype=np.float32)
        phi = np.clip(phi, -5.0, 5.0)
        return phi

    def act(self, obs_dict, eval: bool=False) -> float:
        o = torch.as_tensor(self._encode_obs(obs_dict), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if eval:
                a, _, mu = self.actor(o)
                return float(mu.squeeze(0).cpu().numpy())
            else:
                a, _, _ = self.actor(o)
                return float(a.squeeze(0).cpu().numpy())

    def begin_episode(self):
        pass

    def observe(self, o, a, r, o2, d):
        self.replay.store(o, a, r, o2, d)
        self.total_steps += 1

    def update(self):
        if self.replay.len < self.cfg.batch_size:
            return
        batch = self.replay.sample_batch(self.cfg.batch_size)
        obs = batch['obs']; act = batch['act']; rew = batch['rew']; obs2 = batch['obs2']; done = batch['done']
        alpha = self.log_alpha.exp()

        # Critic update
        with torch.no_grad():
            a2, logp2, _ = self.actor(obs2)
            q1_t = self.q1_t(obs2, a2)
            q2_t = self.q2_t(obs2, a2)
            q_t = torch.min(q1_t, q2_t) - alpha * logp2
            backup = rew + self.cfg.gamma * (1 - done) * q_t
        q1 = self.q1(obs, act)
        q2 = self.q2(obs, act)
        q1_loss = ((q1 - backup)**2).mean()
        q2_loss = ((q2 - backup)**2).mean()
        self.q1_optim.zero_grad(); q1_loss.backward(); self.q1_optim.step()
        self.q2_optim.zero_grad(); q2_loss.backward(); self.q2_optim.step()

        # Actor update
        a, logp, _ = self.actor(obs)
        q1_pi = self.q1(obs, a)
        q2_pi = self.q2(obs, a)
        q_pi = torch.min(q1_pi, q2_pi)
        pi_loss = (alpha * logp - q_pi).mean()
        self.pi_optim.zero_grad(); pi_loss.backward(); self.pi_optim.step()

        # Temperature (alpha) update
        if self.alpha_optim is not None:
            alpha_loss = (-self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad(); alpha_loss.backward(); self.alpha_optim.step()

        # Soft target updates
        with torch.no_grad():
            for p, p_t in zip(self.q1.parameters(), self.q1_t.parameters()):
                p_t.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
            for p, p_t in zip(self.q2.parameters(), self.q2_t.parameters()):
                p_t.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

    def end_episode(self):
        pass

    # --- persistence ---
    def save(self, name: str, out_dir: str = "agents"):
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(out_dir, f"{name}_sac_actor.pth"))
        torch.save(self.q1.state_dict(),    os.path.join(out_dir, f"{name}_sac_q1.pth"))
        torch.save(self.q2.state_dict(),    os.path.join(out_dir, f"{name}_sac_q2.pth"))
        meta = dict(cfg={k: float(getattr(self.cfg, k)) if isinstance(getattr(self.cfg, k), float) else getattr(self.cfg, k)
                          for k in ['u_rl_max','gamma','tau','lr','batch_size','start_steps','updates_per_step','alpha','autotune_alpha']})
        with open(os.path.join(out_dir, f"{name}_sac_meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)

    def load(self, name: str, in_dir: str = "agents"):
        self.actor.load_state_dict(torch.load(os.path.join(in_dir, f"{name}_sac_actor.pth"), map_location='cpu'))
        self.q1.load_state_dict(torch.load(os.path.join(in_dir, f"{name}_sac_q1.pth"), map_location='cpu'))
        self.q2.load_state_dict(torch.load(os.path.join(in_dir, f"{name}_sac_q2.pth"), map_location='cpu'))
        # critics are not needed for pure evaluation but loading keeps coherence
