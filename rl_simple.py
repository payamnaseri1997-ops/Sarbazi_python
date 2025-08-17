# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 01:13:22 2025

@author: elecomp
"""

# ==================================
# === rl_simple.py (SIMPLE agent) ===
# ==================================
"""Simple residual RL policy: linear Gaussian policy trained with REINFORCE.
Provides save()/load() so it can be persisted and reused by name.
"""
from dataclasses import dataclass
import numpy as np
import os, json

class ResidualAgentAPI:  # minimal duck-typing to satisfy main
    def act(self, obs, eval: bool=False) -> float: ...
    def begin_episode(self): ...
    def end_episode(self): ...
    def observe(self, o, a, r, o2, d): ...
    def update(self): ...
    def save(self, name: str, out_dir: str = "agents"): ...

@dataclass
class RLConfig:
    lr: float = 1e-3           # learning rate for weight updates
    sigma: float = 0.1         # exploration std (Nm)
    gamma: float = 0.995       # return discount factor
    u_rl_max: float = 0.16     # residual torque bound (≈20% of SMC torque limit)
    feature_clip: float = 5.0  # clip features to stabilize

class SimpleResidualPolicy(ResidualAgentAPI):
    """Linear policy with Gaussian noise: a ~ N(w^T phi, sigma^2).
    Trained per-episode using REINFORCE (with a value-less baseline: mean return).
    Features: [e, edot, s, omega, 1.0, time_frac] (each clipped to ±feature_clip)
    """
    def __init__(self, n_features: int, cfg: RLConfig):
        self.w = np.zeros(n_features, dtype=np.float32)
        self.cfg = cfg
        # Episode buffers
        self._phi: list[np.ndarray] = []
        self._mu: list[float] = []
        self._a: list[float] = []
        self._r: list[float] = []

    # --- API ---
    def begin_episode(self):
        self._phi.clear(); self._mu.clear(); self._a.clear(); self._r.clear()

    def act(self, obs: dict, eval: bool=False) -> float:
        phi = np.array([obs['e'], obs['edot'], obs['s'], obs['omega'], 1.0, obs['time_frac']], dtype=np.float32)
        phi = np.clip(phi, -self.cfg.feature_clip, self.cfg.feature_clip)
        mu = float(np.dot(self.w, phi))
        mu = float(np.clip(mu, -self.cfg.u_rl_max, self.cfg.u_rl_max))
        if eval:
            a = mu
        else:
            a = float(np.random.randn() * self.cfg.sigma + mu)
            a = float(np.clip(a, -self.cfg.u_rl_max, self.cfg.u_rl_max))
            self._phi.append(phi); self._mu.append(mu); self._a.append(a)
        return a

    def observe(self, o, a, r, o2, d):
        self._r.append(float(r))

    def update(self):
        # no per-step update; do it at episode end
        pass

    def end_episode(self):
        if not self._r:
            return
        # returns
        R = 0.0
        G = np.zeros(len(self._r), dtype=np.float32)
        for t in reversed(range(len(self._r))):
            R = self._r[t] + self.cfg.gamma * R
            G[t] = R
        b = float(G.mean())
        grad = np.zeros_like(self.w)
        var = (self.cfg.sigma ** 2)
        for phi, mu, a, g in zip(self._phi, self._mu, self._a, G):
            adv = g - b
            grad += ((a - mu) / var) * phi * adv
        self.w += self.cfg.lr * grad
        # clear
        self._phi.clear(); self._mu.clear(); self._a.clear(); self._r.clear()

    # --- persistence ---
    def save(self, name: str, out_dir: str = "agents"):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}_simple.npz")
        np.savez(
            path,
            w=self.w,
            cfg=np.array([self.cfg.lr, self.cfg.sigma, self.cfg.gamma, self.cfg.u_rl_max, self.cfg.feature_clip], dtype=np.float32),
        )
        with open(os.path.join(out_dir, f"{name}_simple.readme.txt"), "w") as f:
            f.write("Linear residual policy (REINFORCE). Keys: w, cfg=[lr,sigma,gamma,u_rl_max,feature_clip]")

    def load(self, name: str, in_dir: str = "agents"):
        path = os.path.join(in_dir, f"{name}_simple.npz")
        data = np.load(path, allow_pickle=True)
        self.w = data['w']
        lr, sigma, gamma, u_rl_max, feature_clip = data['cfg'].tolist()
        self.cfg = RLConfig(lr=float(lr), sigma=float(sigma), gamma=float(gamma), u_rl_max=float(u_rl_max), feature_clip=float(feature_clip))
