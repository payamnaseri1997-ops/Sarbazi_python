# -*- coding: utf-8 -*-
"""
sac_gps_lqr_guided_addon.py

Conventional actor-critic RL version of the LQR-guided trajectory optimizer.

This add-on keeps your existing plant dynamics and TDE+SMC controller unchanged.
It changes only the reference trajectory theta_ref(t).

Main idea
---------
Existing LQR/iLQR trajectory is the guide. The SAC actor does not output torque.
It outputs residual trajectory-shape parameters around the LQR guide:

    v_new(z) = v_LQR(z) * exp(B(z) @ p_shape)

The last actor output adjusts duration. The final reference is normalized so:

    theta_ref(0) = 0
    theta_ref(T) = theta_goal

This is conventional deep RL:
    - stochastic actor
    - two critics Q1/Q2
    - target critics
    - entropy temperature alpha
    - replay buffer
    - off-policy SAC updates

The GPS part is that the policy is guided by the LQR trajectory and can also be
warm-started with local LQR-guided CEM teacher trajectories.

Usage
-----
Place this file next to:
    true_gps_lqr_guided_addon.py
    LQR_TrjOPt_TDESMCwithRLresidual.py or LQR_TrjOPt_TDESMCwithRLresidual_2.py

Then run:

from sac_gps_lqr_guided_addon import run_sac_gps_experiment
results = run_sac_gps_experiment()

Fast smoke test:
results = run_sac_gps_experiment(
    train_goal_degs=(30, 90, 150),
    test_goal_degs=(45, 135),
    tilt_degs=(0, 10, 20),
    teacher_cem_iters=2,
    teacher_population=8,
    total_interactions=80,
    updates_per_interaction=4,
    save_dir="sac_gps_smoke",
)
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Sequence
import os, math, json
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    torch = None
    nn = optim = F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    import true_gps_lqr_guided_addon as gps
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "This file requires true_gps_lqr_guided_addon.py in the same folder. "
        f"Import error: {exc}"
    )


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class SACGPSConfig:
    hidden_sizes: Tuple[int, int] = (128, 128)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.0          # one-step contextual MDP: one action = one full trajectory
    tau: float = 0.005
    batch_size: int = 128
    replay_size: int = 200000
    start_random_steps: int = 300
    updates_per_interaction: int = 1
    alpha_init: float = 0.2
    autotune_alpha: bool = True
    target_entropy_scale: float = 1.0

    # normalized action [-1,1] -> trajectory params
    shape_action_scale: float = 1.30
    duration_action_scale: float = 1.20

    # reward = (existing_cost - candidate_cost) / reward_scale
    reward_scale: float = 10000.0
    reward_clip: float = 10.0

    # GPS warm start
    use_teacher_prefill: bool = True
    behavior_clone_epochs: int = 300
    behavior_clone_lr: float = 1e-3

    # optional critic-based action refinement at evaluation
    critic_refine_steps: int = 50
    critic_refine_lr: float = 0.03
    critic_refine_l2: float = 0.002


def mkdir(path: Optional[str]):
    if path:
        os.makedirs(path, exist_ok=True)


def make_default_configs():
    obj_cfg = gps.GPSObjectiveConfig(
        omega_limit=0.2,
        torque_limit=2000.0,
        command_limit=500.0,
        w_energy=1.0,
        w_theta_track=5.0,
        w_omega_track=1.0,
        w_final_theta=5.0e4,
        w_final_omega=5.0e3,
        w_ref_omega_violation=1.0e6,
        w_actual_omega_violation=1.0e6,
        w_tau_violation=1.0e5,
        w_command_violation=1.0e5,
        w_guide_theta=0.05,
        w_guide_omega=0.01,
        w_duration=0.001,
    )
    traj_cfg = gps.GPSTrajectoryConfig(
        n_basis=6,
        omega_ref_limit=0.2,
        duration_margin=1.05,
        max_duration_factor=2.5,
        min_extra_duration=5.0,
        duration_exp_scale=0.35,
        z_grid_size=1201,
        hidden_sizes=(128, 128),
    )
    return traj_cfg, obj_cfg, SACGPSConfig()


def action_scales(traj_cfg: gps.GPSTrajectoryConfig, cfg: SACGPSConfig) -> np.ndarray:
    return np.r_[np.ones(traj_cfg.n_basis) * cfg.shape_action_scale, cfg.duration_action_scale].astype(np.float32)


def action_to_params(action: np.ndarray, traj_cfg: gps.GPSTrajectoryConfig, cfg: SACGPSConfig) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    return (np.clip(a, -1.0, 1.0) * action_scales(traj_cfg, cfg)).astype(float)


def params_to_action(params: np.ndarray, traj_cfg: gps.GPSTrajectoryConfig, cfg: SACGPSConfig) -> np.ndarray:
    p = np.asarray(params, dtype=np.float32).reshape(-1)
    return np.clip(p / action_scales(traj_cfg, cfg), -1.0, 1.0).astype(np.float32)


def case_obs(case: gps.GPSCase) -> np.ndarray:
    return gps.case_features(case).astype(np.float32)


# -----------------------------------------------------------------------------
# Replay buffer
# -----------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.size = int(size)
        self.ptr = 0
        self.len = 0
        self.obs = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.act = np.zeros((self.size, act_dim), dtype=np.float32)
        self.rew = np.zeros((self.size, 1), dtype=np.float32)
        self.obs2 = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.done = np.ones((self.size, 1), dtype=np.float32)

    def store(self, obs, act, rew, obs2=None, done=True):
        i = self.ptr % self.size
        self.obs[i] = np.asarray(obs, dtype=np.float32)
        self.act[i] = np.asarray(act, dtype=np.float32)
        self.rew[i, 0] = float(rew)
        self.obs2[i] = self.obs[i] if obs2 is None else np.asarray(obs2, dtype=np.float32)
        self.done[i, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.size
        self.len = min(self.len + 1, self.size)

    def sample(self, batch_size: int, device):
        idx = np.random.randint(0, self.len, size=int(batch_size))
        return dict(
            obs=torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device),
            act=torch.as_tensor(self.act[idx], dtype=torch.float32, device=device),
            rew=torch.as_tensor(self.rew[idx], dtype=torch.float32, device=device),
            obs2=torch.as_tensor(self.obs2[idx], dtype=torch.float32, device=device),
            done=torch.as_tensor(self.done[idx], dtype=torch.float32, device=device),
        )


# -----------------------------------------------------------------------------
# SAC networks
# -----------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, inp: int, out: int, hidden_sizes: Tuple[int, ...], activation=nn.ReLU):
        super().__init__()
        layers: List[nn.Module] = []
        last = inp
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), activation()]
            last = h
        layers.append(nn.Linear(last, out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        self.net = MLP(obs_dim, 2 * act_dim, hidden_sizes)
        self.log_std_min = -5.0
        self.log_std_max = 1.0

    def forward(self, obs, deterministic: bool = False, with_logprob: bool = True):
        mu_logstd = self.net(obs)
        mu, log_std = torch.chunk(mu_logstd, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        z = mu if deterministic else dist.rsample()
        action = torch.tanh(z)
        if with_logprob:
            logp = dist.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
            logp = logp.sum(dim=-1, keepdim=True)
        else:
            logp = None
        return action, logp, torch.tanh(mu)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        self.q = MLP(obs_dim + act_dim, 1, hidden_sizes)

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1))


class SACGPSAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: SACGPSConfig):
        if torch is None:
            raise RuntimeError(f"PyTorch import failed: {_TORCH_IMPORT_ERROR}")
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.actor = SquashedGaussianActor(obs_dim, act_dim, cfg.hidden_sizes).to(self.device)
        self.q1 = Critic(obs_dim, act_dim, cfg.hidden_sizes).to(self.device)
        self.q2 = Critic(obs_dim, act_dim, cfg.hidden_sizes).to(self.device)
        self.q1_t = Critic(obs_dim, act_dim, cfg.hidden_sizes).to(self.device)
        self.q2_t = Critic(obs_dim, act_dim, cfg.hidden_sizes).to(self.device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.pi_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)
        self.replay = ReplayBuffer(obs_dim, act_dim, cfg.replay_size)
        self.total_interactions = 0
        if cfg.autotune_alpha:
            self.log_alpha = torch.tensor(math.log(cfg.alpha_init), dtype=torch.float32, requires_grad=True, device=self.device)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
            self.target_entropy = -float(act_dim) * cfg.target_entropy_scale
        else:
            self.log_alpha = torch.tensor(math.log(cfg.alpha_init), dtype=torch.float32, requires_grad=False, device=self.device)
            self.alpha_opt = None
            self.target_entropy = None

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a, _, mu = self.actor(obs_t, deterministic=deterministic, with_logprob=False)
            out = mu if deterministic else a
        return out.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self, n_updates: int = 1) -> Dict[str, float]:
        if self.replay.len < self.cfg.batch_size:
            return {}
        logs: Dict[str, float] = {}
        for _ in range(int(n_updates)):
            batch = self.replay.sample(self.cfg.batch_size, self.device)
            obs, act, rew, obs2, done = batch["obs"], batch["act"], batch["rew"], batch["obs2"], batch["done"]
            alpha = self.alpha
            with torch.no_grad():
                a2, logp2, _ = self.actor(obs2, deterministic=False, with_logprob=True)
                q_t = torch.min(self.q1_t(obs2, a2), self.q2_t(obs2, a2)) - alpha * logp2
                backup = rew + self.cfg.gamma * (1.0 - done) * q_t
            q1_loss = F.mse_loss(self.q1(obs, act), backup)
            q2_loss = F.mse_loss(self.q2(obs, act), backup)
            self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
            self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()
            a_pi, logp_pi, _ = self.actor(obs, deterministic=False, with_logprob=True)
            q_pi = torch.min(self.q1(obs, a_pi), self.q2(obs, a_pi))
            pi_loss = (alpha.detach() * logp_pi - q_pi).mean()
            self.pi_opt.zero_grad(); pi_loss.backward(); self.pi_opt.step()
            if self.alpha_opt is not None:
                alpha_loss = (-self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
                self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
            else:
                alpha_loss = torch.tensor(0.0)
            with torch.no_grad():
                for p, pt in zip(self.q1.parameters(), self.q1_t.parameters()):
                    pt.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)
                for p, pt in zip(self.q2.parameters(), self.q2_t.parameters()):
                    pt.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)
            logs = dict(
                q1_loss=float(q1_loss.detach().cpu().item()),
                q2_loss=float(q2_loss.detach().cpu().item()),
                pi_loss=float(pi_loss.detach().cpu().item()),
                alpha=float(self.alpha.detach().cpu().item()),
                alpha_loss=float(alpha_loss.detach().cpu().item()),
            )
        return logs

    def behavior_clone(self, obs_arr: np.ndarray, act_arr: np.ndarray, epochs: int, lr: float) -> List[float]:
        if epochs <= 0 or len(obs_arr) == 0:
            return []
        opt = optim.Adam(self.actor.parameters(), lr=lr)
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(act_arr, dtype=torch.float32, device=self.device)
        losses: List[float] = []
        for ep in range(int(epochs)):
            _, _, mu = self.actor(obs_t, deterministic=True, with_logprob=False)
            loss = F.mse_loss(mu, act_t)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu().item()))
            if (ep + 1) % max(1, epochs // 4) == 0:
                print(f"behavior clone {ep+1}/{epochs}: loss={losses[-1]:.6g}")
        return losses

    def save(self, path: str):
        mkdir(os.path.dirname(path) or ".")
        torch.save(dict(
            actor=self.actor.state_dict(), q1=self.q1.state_dict(), q2=self.q2.state_dict(),
            q1_t=self.q1_t.state_dict(), q2_t=self.q2_t.state_dict(),
            log_alpha=float(self.log_alpha.detach().cpu().item()), cfg=asdict(self.cfg)
        ), path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        if "q1_t" in ckpt:
            self.q1_t.load_state_dict(ckpt["q1_t"])
        else:
            self.q1_t.load_state_dict(self.q1.state_dict())
        if "q2_t" in ckpt:
            self.q2_t.load_state_dict(ckpt["q2_t"])
        else:
            self.q2_t.load_state_dict(self.q2.state_dict())
        if "log_alpha" in ckpt:
            with torch.no_grad():
                self.log_alpha.copy_(torch.tensor(float(ckpt["log_alpha"]), dtype=torch.float32, device=self.device))
        print(f"Loaded SAC-GPS agent from: {path}")


# -----------------------------------------------------------------------------
# Environment/evaluation helpers
# -----------------------------------------------------------------------------

class BaselineCache:
    def __init__(self, obj_cfg: gps.GPSObjectiveConfig):
        self.obj_cfg = obj_cfg
        self.cache: Dict[Tuple[float, float, float], Tuple[Dict[str, float], Dict[str, np.ndarray], np.ndarray, float]] = {}

    @staticmethod
    def key(case: gps.GPSCase):
        return (round(case.theta_goal, 12), round(case.alpha, 12), round(case.phi, 12))

    def get(self, case: gps.GPSCase, seed: int = 0):
        k = self.key(case)
        if k not in self.cache:
            self.cache[k] = gps.evaluate_existing_case(case, self.obj_cfg, seed=seed)
        return self.cache[k]


def evaluate_action(case, action, traj_cfg, obj_cfg, sac_cfg, baseline_cache, seed=0) -> Dict[str, object]:
    params = action_to_params(action, traj_cfg, sac_cfg)
    existing_metrics, existing_logs, existing_ref, existing_T = baseline_cache.get(case, seed=seed)
    try:
        metrics, logs, theta_ref, T, extra, base_ref, base_T = gps.evaluate_guided_params(
            case, params, traj_cfg, obj_cfg, seed=seed
        )
        cost = float(metrics["total"])
        failed = False
    except Exception:
        metrics, logs, theta_ref, T, extra, base_ref, base_T = None, None, None, None, None, existing_ref, existing_T
        cost = float(existing_metrics["total"] + 1e8)
        failed = True
    raw_reward = (float(existing_metrics["total"]) - cost) / max(sac_cfg.reward_scale, 1e-12)
    reward = float(np.clip(raw_reward, -sac_cfg.reward_clip, sac_cfg.reward_clip))
    return dict(
        case=case, action=np.asarray(action, dtype=np.float32), params=params,
        reward=reward, raw_reward=float(raw_reward), cost=cost,
        existing_cost=float(existing_metrics["total"]),
        improvement_pct=100.0 * (float(existing_metrics["total"]) - cost) / max(abs(float(existing_metrics["total"])), 1e-12),
        metrics=metrics, logs=logs, theta_ref=theta_ref, T=T, extra=extra,
        existing_metrics=existing_metrics, existing_logs=existing_logs, existing_ref=existing_ref, existing_T=existing_T,
        base_ref=base_ref, base_T=base_T, failed=failed
    )


def critic_refine_action(agent: SACGPSAgent, obs: np.ndarray, action_init: np.ndarray, steps: int, lr: float, l2: float) -> np.ndarray:
    if steps <= 0:
        return np.asarray(action_init, dtype=np.float32)
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
    a0 = np.clip(np.asarray(action_init, dtype=np.float32), -0.999, 0.999)
    y = torch.tensor(np.arctanh(a0), dtype=torch.float32, device=agent.device, requires_grad=True)
    opt = optim.Adam([y], lr=lr)
    a0_t = torch.as_tensor(a0, dtype=torch.float32, device=agent.device).unsqueeze(0)
    for _ in range(int(steps)):
        a = torch.tanh(y).unsqueeze(0)
        q = torch.min(agent.q1(obs_t, a), agent.q2(obs_t, a))
        loss = -(q.mean() - l2 * torch.mean((a - a0_t) ** 2))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return torch.tanh(y).cpu().numpy().astype(np.float32)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def _teacher_key(goal_deg: float, alpha_deg: float, phi_deg: float) -> Tuple[float, float, float]:
    return (round(float(goal_deg), 6), round(float(alpha_deg), 6), round(float(phi_deg), 6))


def _load_teacher_summary(save_dir: Optional[str]) -> Dict[Tuple[float, float, float], Dict[str, object]]:
    """Load saved teachers without rerunning CEM.

    Supported formats:
    1. ``teacher_summary.json`` produced by this SAC-GPS file.
       Rows contain: goal, alpha, phi, action.

    2. ``training_teachers.json`` produced by ``true_gps_lqr_guided_addon.py``.
       Rows contain: theta_goal_deg, alpha_deg, phi_deg, best_params.

    The second one is the file you already generated when you ran the true-GPS
    local teacher stage. Loading it prevents messages such as:
        === Local GPS teacher ... ===
        CEM ... iter ...
    """
    if not save_dir:
        return {}

    candidates = [
        os.path.join(save_dir, "teacher_summary.json"),
        os.path.join(save_dir, "training_teachers.json"),
    ]

    path = next((q for q in candidates if os.path.exists(q)), None)
    if path is None:
        return {}

    with open(path, "r") as f:
        rows = json.load(f)

    out = {}
    base = os.path.basename(path)
    for row in rows:
        if base == "teacher_summary.json":
            # SAC-GPS format: already contains normalized action.
            key = _teacher_key(row["goal"], row["alpha"], row["phi"])
            row = dict(row)
            row["_format"] = "action"
            out[key] = row
        else:
            # true-GPS format: contains raw trajectory parameters.
            key = _teacher_key(row["theta_goal_deg"], row["alpha_deg"], row["phi_deg"])
            row = dict(row)
            row["_format"] = "params"
            out[key] = row

    print(f"Loaded {len(out)} saved teacher rows from: {path}")
    return out


def prefill_from_saved_teachers(agent, train_cases, traj_cfg, obj_cfg, sac_cfg, baseline_cache, save_dir, seed=0):
    """Prefill replay buffer and behavior-clone actor from saved teacher_summary.json.

    It evaluates the saved teacher actions once to reconstruct rewards/logs in
    the current session, but it does NOT run the CEM teacher optimizer.
    """
    teacher_map = _load_teacher_summary(save_dir)
    if not teacher_map:
        return None

    rows, bc_obs, bc_act = [], [], []
    zero_action = np.zeros(traj_cfg.n_basis + 1, dtype=np.float32)

    for i, case in enumerate(train_cases):
        obs = case_obs(case)

        zero_ev = evaluate_action(case, zero_action, traj_cfg, obj_cfg, sac_cfg, baseline_cache, seed=seed+i)
        agent.replay.store(obs, zero_action, zero_ev["reward"], obs2=obs, done=True)

        key = _teacher_key(case.theta_goal_deg, case.alpha_deg, case.phi_deg)
        saved = teacher_map.get(key)
        if saved is None:
            print(f"No saved teacher for {case.label()} -- skipped teacher action for this case.")
            continue

        if saved.get("_format") == "action":
            best_action = np.asarray(saved["action"], dtype=np.float32)
        elif saved.get("_format") == "params":
            best_params = np.asarray(saved["best_params"], dtype=np.float32)
            best_action = params_to_action(best_params, traj_cfg, sac_cfg)
        else:
            raise ValueError(f"Unknown saved teacher format for {case.label()}: {saved.get('_format')}")

        best_ev = evaluate_action(case, best_action, traj_cfg, obj_cfg, sac_cfg, baseline_cache, seed=seed+2000+i)

        agent.replay.store(obs, best_action, best_ev["reward"], obs2=obs, done=True)
        bc_obs.append(obs)
        bc_act.append(best_action)
        rows.append(dict(case=case, zero_eval=zero_ev, best_eval=best_ev, best_action=best_action, saved_teacher=saved))

        print(
            f"saved teacher {i+1}/{len(train_cases)}: {case.label()} | "
            f"existing={best_ev['existing_cost']:.6g}, cost={best_ev['cost']:.6g}, "
            f"improvement={best_ev['improvement_pct']:.2f}%"
        )

    bc_losses = []
    if bc_obs and sac_cfg.behavior_clone_epochs > 0:
        bc_losses = agent.behavior_clone(
            np.stack(bc_obs),
            np.stack(bc_act),
            sac_cfg.behavior_clone_epochs,
            sac_cfg.behavior_clone_lr,
        )

    return dict(rows=rows, bc_losses=bc_losses, loaded_from="saved_teacher_json")


def prefill_with_gps_teachers(agent, train_cases, traj_cfg, obj_cfg, sac_cfg, baseline_cache, teacher_cem_iters, teacher_population, seed, save_dir=None):
    rows, bc_obs, bc_act = [], [], []
    zero_action = np.zeros(traj_cfg.n_basis + 1, dtype=np.float32)
    for i, case in enumerate(train_cases):
        obs = case_obs(case)
        zero_ev = evaluate_action(case, zero_action, traj_cfg, obj_cfg, sac_cfg, baseline_cache, seed=seed+i)
        agent.replay.store(obs, zero_action, zero_ev["reward"], obs2=obs, done=True)
        print(f"\n=== LQR-guided local teacher {i+1}/{len(train_cases)}: {case.label()} ===")
        teacher = gps.cem_optimize_guided_case(
            case=case, traj_cfg=traj_cfg, obj_cfg=obj_cfg,
            init_mean=np.zeros(traj_cfg.n_basis + 1),
            init_std=np.r_[np.ones(traj_cfg.n_basis) * 0.55, 0.35],
            cem_iters=teacher_cem_iters, population=teacher_population,
            elite_frac=0.25, seed=seed + 1000 + 31 * i,
        )
        best_params = np.asarray(teacher["best"]["params"], dtype=np.float32)
        best_action = params_to_action(best_params, traj_cfg, sac_cfg)
        best_ev = evaluate_action(case, best_action, traj_cfg, obj_cfg, sac_cfg, baseline_cache, seed=seed+2000+i)
        agent.replay.store(obs, best_action, best_ev["reward"], obs2=obs, done=True)
        bc_obs.append(obs); bc_act.append(best_action)
        rows.append(dict(case=case, zero_eval=zero_ev, teacher=teacher, best_eval=best_ev, best_action=best_action, best_params=best_params))
        print(f"teacher: existing={best_ev['existing_cost']:.6g}, cost={best_ev['cost']:.6g}, improvement={best_ev['improvement_pct']:.2f}%")
    bc_losses = []
    if bc_obs:
        bc_losses = agent.behavior_clone(np.stack(bc_obs), np.stack(bc_act), sac_cfg.behavior_clone_epochs, sac_cfg.behavior_clone_lr)
    if save_dir:
        with open(os.path.join(save_dir, "teacher_summary.json"), "w") as f:
            json.dump([dict(goal=r["case"].theta_goal_deg, alpha=r["case"].alpha_deg, phi=r["case"].phi_deg,
                            existing=r["best_eval"]["existing_cost"], teacher=r["best_eval"]["cost"],
                            improvement_pct=r["best_eval"]["improvement_pct"], action=r["best_action"].tolist()) for r in rows], f, indent=2)
    return dict(rows=rows, bc_losses=bc_losses, loaded_from="fresh_cem")


def _load_history_if_available(save_dir: Optional[str]) -> List[Dict[str, float]]:
    if not save_dir:
        return []
    path = os.path.join(save_dir, "training_history.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        hist = json.load(f)
    print(f"Loaded existing training history with {len(hist)} records from: {path}")
    return hist


def train_sac_gps_agent(
    train_cases,
    traj_cfg,
    obj_cfg,
    sac_cfg,
    total_interactions=1000,
    teacher_cem_iters=4,
    teacher_population=20,
    eval_every=50,
    seed=0,
    save_dir=None,
    use_saved_teacher_summary=True,
    allow_fresh_cem_teachers=False,
    load_existing_agent=True,
    skip_training_if_agent_loaded=True,
):
    """Train/evaluate SAC-GPS while reusing saved files when possible.

    Resume behavior:
    1. If save_dir/sac_gps_agent.pt exists, load it.
    2. If skip_training_if_agent_loaded=True, skip training and evaluate only.
    3. If no checkpoint is loaded, use teacher_summary.json to prefill replay
       and behavior-clone instead of running CEM.
    4. Fresh CEM teachers run only if allow_fresh_cem_teachers=True.
    """
    if torch is None:
        raise RuntimeError(f"PyTorch import failed: {_TORCH_IMPORT_ERROR}")

    mkdir(save_dir)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = SACGPSAgent(obs_dim=3, act_dim=traj_cfg.n_basis + 1, cfg=sac_cfg)
    baseline_cache = BaselineCache(obj_cfg)
    history: List[Dict[str, float]] = _load_history_if_available(save_dir)
    teacher_output = None

    checkpoint_path = os.path.join(save_dir, "sac_gps_agent.pt") if save_dir else "sac_gps_agent.pt"
    agent_loaded = False

    if load_existing_agent and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        agent_loaded = True

    if agent_loaded and skip_training_if_agent_loaded:
        print("Existing SAC-GPS checkpoint loaded. Skipping training and using saved agent for evaluation.")
        if save_dir:
            actor_path = os.path.join(save_dir, "sac_gps_actor.pt")
            if not os.path.exists(actor_path):
                torch.save(agent.actor.state_dict(), actor_path)
                print(f"Exported actor-only checkpoint to: {actor_path}")
        return dict(agent=agent, history=history, teacher_output=None, baseline_cache=baseline_cache, agent_loaded=True)

    if sac_cfg.use_teacher_prefill:
        if use_saved_teacher_summary:
            teacher_output = prefill_from_saved_teachers(
                agent, train_cases, traj_cfg, obj_cfg, sac_cfg, baseline_cache, save_dir, seed
            )

        if teacher_output is None:
            if allow_fresh_cem_teachers:
                teacher_output = prefill_with_gps_teachers(
                    agent, train_cases, traj_cfg, obj_cfg, sac_cfg, baseline_cache,
                    teacher_cem_iters, teacher_population, seed, save_dir
                )
            else:
                print("No saved teacher_summary.json found and fresh CEM teachers are disabled.")
                print("Continuing without teacher prefill. Set allow_fresh_cem_teachers=True to rebuild teachers.")

    for step in range(1, int(total_interactions) + 1):
        case = train_cases[int(rng.integers(0, len(train_cases)))]
        obs = case_obs(case)

        if agent.total_interactions < sac_cfg.start_random_steps:
            action = rng.uniform(-1.0, 1.0, size=traj_cfg.n_basis + 1).astype(np.float32)
        else:
            action = agent.act(obs, deterministic=False)

        ev = evaluate_action(case, action, traj_cfg, obj_cfg, sac_cfg, baseline_cache, seed=seed+10000+step)
        agent.replay.store(obs, action, ev["reward"], obs2=obs, done=True)
        agent.total_interactions += 1

        upd = agent.update(sac_cfg.updates_per_interaction)

        if step % max(1, eval_every) == 0 or step == 1:
            eval_rows = evaluate_sac_gps_policy(
                agent, train_cases[:min(8, len(train_cases))], traj_cfg, obj_cfg, sac_cfg,
                baseline_cache, seed=seed+50000+step, critic_refine=False
            )
            mean_existing = float(np.mean([r["existing_metrics"]["total"] for r in eval_rows]))
            mean_actor = float(np.mean([r["actor_eval"]["cost"] for r in eval_rows]))
            mean_imp = 100.0 * (mean_existing - mean_actor) / max(abs(mean_existing), 1e-12)
            rec = dict(
                step=step,
                train_reward=float(ev["reward"]),
                train_raw_reward=float(ev["raw_reward"]),
                train_cost=float(ev["cost"]),
                train_existing_cost=float(ev["existing_cost"]),
                mean_eval_existing=mean_existing,
                mean_eval_actor=mean_actor,
                mean_eval_improvement_pct=mean_imp,
                buffer=float(agent.replay.len),
                alpha=float(agent.alpha.detach().cpu().item()),
            )
            rec.update({k: float(v) for k, v in upd.items()})
            history.append(rec)
            print(
                f"SAC-GPS step {step:05d}/{total_interactions}: "
                f"train_reward={ev['reward']:.4f}, actor_eval={mean_actor:.6g}, "
                f"existing_eval={mean_existing:.6g}, improvement={mean_imp:.2f}%, "
                f"buffer={agent.replay.len}"
            )

    if save_dir:
        agent_path = os.path.join(save_dir, "sac_gps_agent.pt")
        actor_path = os.path.join(save_dir, "sac_gps_actor.pt")
        agent.save(agent_path)
        # Actor-only export for LQR_TrjOPt_TDESMCwithRLresidual.py inference.
        # The main file should never train; it only loads this actor and decodes theta_ref(t).
        torch.save(agent.actor.state_dict(), actor_path)
        with open(os.path.join(save_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        with open(os.path.join(save_dir, "configs.json"), "w") as f:
            json.dump(dict(sac_cfg=asdict(sac_cfg), traj_cfg=asdict(traj_cfg), obj_cfg=asdict(obj_cfg)), f, indent=2)

    return dict(agent=agent, history=history, teacher_output=teacher_output, baseline_cache=baseline_cache, agent_loaded=agent_loaded)


# -----------------------------------------------------------------------------
# Evaluation + plots
# -----------------------------------------------------------------------------

def evaluate_sac_gps_policy(agent, test_cases, traj_cfg, obj_cfg, sac_cfg, baseline_cache=None, seed=0, critic_refine=True):
    baseline_cache = baseline_cache or BaselineCache(obj_cfg)
    rows: List[Dict[str, object]] = []
    zero_action = np.zeros(traj_cfg.n_basis + 1, dtype=np.float32)
    for i, case in enumerate(test_cases):
        obs = case_obs(case)
        existing_metrics, existing_logs, existing_ref, existing_T = baseline_cache.get(case, seed=seed+i)
        actor_action = agent.act(obs, deterministic=True)
        actor_ev = evaluate_action(case, actor_action, traj_cfg, obj_cfg, sac_cfg, baseline_cache, seed=seed+1000+i)
        if critic_refine:
            refined_action = critic_refine_action(agent, obs, actor_action, sac_cfg.critic_refine_steps, sac_cfg.critic_refine_lr, sac_cfg.critic_refine_l2)
            refined_ev = evaluate_action(case, refined_action, traj_cfg, obj_cfg, sac_cfg, baseline_cache, seed=seed+2000+i)
        else:
            refined_action = actor_action.copy(); refined_ev = actor_ev
        zero_ev = evaluate_action(case, zero_action, traj_cfg, obj_cfg, sac_cfg, baseline_cache, seed=seed+3000+i)
        best_ev = min([actor_ev, refined_ev, zero_ev], key=lambda d: d["cost"])
        rows.append(dict(case=case, existing_metrics=existing_metrics, existing_logs=existing_logs,
                         existing_ref=existing_ref, existing_T=existing_T, zero_eval=zero_ev,
                         actor_eval=actor_ev, refined_eval=refined_ev, best_eval=best_ev,
                         actor_metrics=actor_ev["metrics"], refined_metrics=refined_ev["metrics"], best_metrics=best_ev["metrics"]))
        print(f"eval {case.label()}: existing={existing_metrics['total']:.6g}, actor={actor_ev['cost']:.6g}, refined={refined_ev['cost']:.6g}, best={best_ev['cost']:.6g}, best_imp={best_ev['improvement_pct']:.2f}%")
    return rows


def _save_or_show(fig, save_dir, name, show):
    if save_dir:
        mkdir(save_dir); fig.savefig(os.path.join(save_dir, name), dpi=180, bbox_inches="tight")
    if show: plt.show()
    else: plt.close(fig)


def plot_training_history(history, save_dir=None, show=True):
    if not history: return
    keys = history[0].keys()
    h = {k: np.array([row.get(k, np.nan) for row in history], dtype=float) for k in keys}
    step = h["step"]
    fig = plt.figure(figsize=(8,5)); plt.plot(step,h["train_reward"],label="train reward"); plt.plot(step,h["train_raw_reward"],label="raw improvement reward"); plt.xlabel("interactions"); plt.ylabel("reward"); plt.title("SAC-GPS reward"); plt.grid(True,alpha=.4); plt.legend(); _save_or_show(fig,save_dir,"training_reward.png",show)
    fig = plt.figure(figsize=(8,5)); plt.plot(step,h["mean_eval_existing"],label="existing LQR cost"); plt.plot(step,h["mean_eval_actor"],label="SAC actor cost"); plt.xlabel("interactions"); plt.ylabel("objective cost"); plt.title("Evaluation cost during training"); plt.grid(True,alpha=.4); plt.legend(); _save_or_show(fig,save_dir,"training_eval_cost.png",show)
    fig = plt.figure(figsize=(8,5)); plt.plot(step,h["mean_eval_improvement_pct"],label="actor improvement %"); plt.axhline(0,linestyle="--",label="no improvement"); plt.xlabel("interactions"); plt.ylabel("% over existing"); plt.title("Improvement over LQR guide"); plt.grid(True,alpha=.4); plt.legend(); _save_or_show(fig,save_dir,"training_improvement.png",show)
    fig = plt.figure(figsize=(8,5));
    for k in ("q1_loss","q2_loss","pi_loss"):
        if k in h: plt.plot(step,h[k],label=k)
    plt.xlabel("interactions"); plt.ylabel("loss"); plt.title("Actor-critic losses"); plt.grid(True,alpha=.4); plt.legend(); _save_or_show(fig,save_dir,"training_losses.png",show)
    fig = plt.figure(figsize=(8,5)); plt.plot(step,h["buffer"],label="buffer size"); plt.plot(step,h["alpha"],label="entropy alpha"); plt.xlabel("interactions"); plt.title("Replay buffer and entropy alpha"); plt.grid(True,alpha=.4); plt.legend(); _save_or_show(fig,save_dir,"training_buffer_alpha.png",show)


def plot_evaluation_summary(rows, obj_cfg, save_dir=None, show=True):
    if not rows: return
    labels = [f"{r['case'].theta_goal_deg:.0f}/{r['case'].alpha_deg:.0f}" for r in rows]
    x = np.arange(len(rows))
    existing = np.array([r["existing_metrics"]["total"] for r in rows], float)
    actor = np.array([r["actor_eval"]["cost"] for r in rows], float)
    refined = np.array([r["refined_eval"]["cost"] for r in rows], float)
    best = np.array([r["best_eval"]["cost"] for r in rows], float)
    fig = plt.figure(figsize=(max(9,.45*len(rows)),5)); plt.plot(x,existing,marker="o",label="existing LQR"); plt.plot(x,actor,marker="o",label="SAC actor"); plt.plot(x,refined,marker="o",label="critic-refined SAC"); plt.plot(x,best,marker="o",label="best safe trajectory"); plt.xticks(x,labels,rotation=60,ha="right"); plt.ylabel("objective cost"); plt.title("Trajectory objective comparison"); plt.grid(True,alpha=.4); plt.legend(); _save_or_show(fig,save_dir,"eval_cost_comparison.png",show)
    imp = 100*(existing-best)/np.maximum(np.abs(existing),1e-12); fig = plt.figure(figsize=(max(9,.45*len(rows)),5)); plt.bar(x,imp,label="best improvement %"); plt.axhline(0,linestyle="--"); plt.xticks(x,labels,rotation=60,ha="right"); plt.ylabel("improvement [%]"); plt.title("SAC-GPS improvement over LQR guide"); plt.grid(True,axis="y",alpha=.4); plt.legend(); _save_or_show(fig,save_dir,"eval_improvement_percent.png",show)
    fig = plt.figure(figsize=(max(9,.45*len(rows)),5)); plt.plot(x,[r["existing_metrics"]["max_abs_omega"] for r in rows],marker="o",label="existing max |omega|"); plt.plot(x,[r["best_eval"]["metrics"]["max_abs_omega"] for r in rows],marker="o",label="SAC-GPS max |omega|"); plt.axhline(obj_cfg.omega_limit,linestyle="--",label="omega limit"); plt.xticks(x,labels,rotation=60,ha="right"); plt.ylabel("rad/s"); plt.title("Velocity constraint"); plt.grid(True,alpha=.4); plt.legend(); _save_or_show(fig,save_dir,"eval_velocity_constraint.png",show)
    fig = plt.figure(figsize=(max(9,.45*len(rows)),5)); plt.plot(x,[r["existing_metrics"]["max_abs_tau_m"] for r in rows],marker="o",label="existing max |tau_m|"); plt.plot(x,[r["best_eval"]["metrics"]["max_abs_tau_m"] for r in rows],marker="o",label="SAC-GPS max |tau_m|"); plt.axhline(obj_cfg.torque_limit,linestyle="--",label="torque limit"); plt.xticks(x,labels,rotation=60,ha="right"); plt.ylabel("N.m"); plt.title("Shaft torque constraint"); plt.grid(True,alpha=.4); plt.legend(); _save_or_show(fig,save_dir,"eval_torque_constraint.png",show)


def plot_case_comparison(row, obj_cfg, save_dir=None, show=True, name_prefix="case"):
    case = row["case"]; e = row["existing_logs"]; b = row["best_eval"]["logs"]
    if b is None: return
    t0=np.asarray(e["t"]); t1=np.asarray(b["t"])
    fig,axes=plt.subplots(5,1,figsize=(10,13),sharex=False); fig.suptitle(f"SAC-GPS vs existing: {case.label()}")
    axes[0].plot(t0,e["theta_ref"],"--",label="existing theta_ref"); axes[0].plot(t0,e["theta"],label="existing theta"); axes[0].plot(t1,b["theta_ref"],"--",label="SAC-GPS theta_ref"); axes[0].plot(t1,b["theta"],label="SAC-GPS theta"); axes[0].axhline(case.theta_goal,linestyle=":",label="theta_goal"); axes[0].set_ylabel("theta [rad]"); axes[0].legend(); axes[0].grid(True,alpha=.4)
    axes[1].plot(t0,np.asarray(e["theta"])-np.asarray(e["theta_ref"]),label="existing theta error"); axes[1].plot(t1,np.asarray(b["theta"])-np.asarray(b["theta_ref"]),label="SAC-GPS theta error"); axes[1].set_ylabel("theta error [rad]"); axes[1].legend(); axes[1].grid(True,alpha=.4)
    axes[2].plot(t0,e["omega"],label="existing omega"); axes[2].plot(t0,e["omega_ref"],"--",label="existing omega_ref"); axes[2].plot(t1,b["omega"],label="SAC-GPS omega"); axes[2].plot(t1,b["omega_ref"],"--",label="SAC-GPS omega_ref"); axes[2].axhline(obj_cfg.omega_limit,linestyle=":",label="+omega limit"); axes[2].axhline(-obj_cfg.omega_limit,linestyle=":",label="-omega limit"); axes[2].set_ylabel("omega [rad/s]"); axes[2].legend(); axes[2].grid(True,alpha=.4)
    axes[3].plot(t0,e["u_total"],label="existing u_total"); axes[3].plot(t1,b["u_total"],label="SAC-GPS u_total"); axes[3].axhline(obj_cfg.command_limit,linestyle=":",label="+command limit"); axes[3].axhline(-obj_cfg.command_limit,linestyle=":",label="-command limit"); axes[3].set_ylabel("command [N.m]"); axes[3].legend(); axes[3].grid(True,alpha=.4)
    tau0=e.get("tau_m",e["u_total"]); tau1=b.get("tau_m",b["u_total"]); axes[4].plot(t0,tau0,label="existing tau_m"); axes[4].plot(t1,tau1,label="SAC-GPS tau_m"); axes[4].axhline(obj_cfg.torque_limit,linestyle=":",label="+torque limit"); axes[4].axhline(-obj_cfg.torque_limit,linestyle=":",label="-torque limit"); axes[4].set_ylabel("tau_m [N.m]"); axes[4].set_xlabel("time [s]"); axes[4].legend(); axes[4].grid(True,alpha=.4)
    fig.tight_layout(rect=[0,0,1,.97]); safe=f"{name_prefix}_g{case.theta_goal_deg:.0f}_a{case.alpha_deg:.0f}_p{case.phi_deg:.0f}.png".replace(".","p"); _save_or_show(fig,save_dir,safe,show)


def print_constraint_report(rows, obj_cfg):
    print("\n=== Constraint / improvement report ===")
    for r in rows:
        case=r["case"]; best=r["best_eval"]; m=best["metrics"]
        print(f"{case.label()}: existing={best['existing_cost']:.6g}, best={best['cost']:.6g}, imp={best['improvement_pct']:.2f}%, max|omega|={m['max_abs_omega']:.4g}/{obj_cfg.omega_limit}, max|tau_m|={m['max_abs_tau_m']:.4g}/{obj_cfg.torque_limit}, max|u|={m['max_abs_u_total']:.4g}/{obj_cfg.command_limit}, final_err={m['final_theta_error']:.4g}")


# -----------------------------------------------------------------------------
# Script Runner — resume from saved files when available
# -----------------------------------------------------------------------------

# This section is intentionally NOT wrapped in run_sac_gps_experiment().
# Run it line-by-line in Spyder.

# 1) Experiment settings
train_goal_degs = (0, 30, 60, 90, 120, 150, 180)
test_goal_degs = (15, 45, 75, 105, 135, 165, 180)
tilt_degs = (0, 5, 10, 15, 20)
coupled_tilts = True

# If a trained agent exists, training can be skipped and only evaluation/plots run.
save_dir = "true_gps_results_smoke"  # folder containing training_teachers.json from true_gps_lqr_guided_addon.py
show_plots = True
seed = 0

# Resume/reuse controls
load_existing_agent_if_available = True
skip_training_if_agent_loaded = True
use_saved_teacher_summary = True
allow_fresh_cem_teachers = False  # keep False to avoid rerunning Local GPS teacher / CEM  # set True only if you really want to rebuild CEM teachers

# Used only if training actually runs.
total_interactions = 1000
teacher_cem_iters = 4
teacher_population = 20
eval_every = 50
updates_per_interaction = None


# 2) Create configs
traj_cfg, obj_cfg, sac_cfg = make_default_configs()

if updates_per_interaction is not None:
    sac_cfg.updates_per_interaction = int(updates_per_interaction)

mkdir(save_dir)


# 3) Create train/test cases
train_cases = gps.make_cases(train_goal_degs, tilt_degs, coupled_tilts=coupled_tilts)
test_cases = gps.make_cases(test_goal_degs, tilt_degs, coupled_tilts=coupled_tilts)


# 4) Print experiment info
print("\n=== SAC-GPS actor-critic trajectory learning/resume ===")
print(
    f"train cases={len(train_cases)}, "
    f"test cases={len(test_cases)}, "
    f"tilts={list(tilt_degs)}, "
    f"coupled={coupled_tilts}"
)
print("TDE+SMC controller and plant dynamics are unchanged.")
print("RL changes only theta_ref(t).")
print(f"save_dir = {save_dir}")


# 5) Load saved SAC agent if available, otherwise train using saved teachers if available
train_output = train_sac_gps_agent(
    train_cases=train_cases,
    traj_cfg=traj_cfg,
    obj_cfg=obj_cfg,
    sac_cfg=sac_cfg,
    total_interactions=total_interactions,
    teacher_cem_iters=teacher_cem_iters,
    teacher_population=teacher_population,
    eval_every=eval_every,
    seed=seed,
    save_dir=save_dir,
    use_saved_teacher_summary=use_saved_teacher_summary,
    allow_fresh_cem_teachers=allow_fresh_cem_teachers,
    load_existing_agent=load_existing_agent_if_available,
    skip_training_if_agent_loaded=skip_training_if_agent_loaded,
)

agent = train_output["agent"]
baseline_cache = train_output["baseline_cache"]


# 6) Evaluate trained/loaded SAC-GPS policy
rows = evaluate_sac_gps_policy(
    agent=agent,
    test_cases=test_cases,
    traj_cfg=traj_cfg,
    obj_cfg=obj_cfg,
    sac_cfg=sac_cfg,
    baseline_cache=baseline_cache,
    seed=seed + 200000,
    critic_refine=True,
)


# 7) Print constraint/improvement report
print_constraint_report(rows, obj_cfg)


# 8) Save evaluation table
table = []

for r in rows:
    case = r["case"]
    best = r["best_eval"]
    m = best["metrics"]

    table.append(
        dict(
            theta_goal_deg=case.theta_goal_deg,
            alpha_deg=case.alpha_deg,
            phi_deg=case.phi_deg,
            existing_cost=best["existing_cost"],
            actor_cost=r["actor_eval"]["cost"],
            refined_cost=r["refined_eval"]["cost"],
            best_cost=best["cost"],
            improvement_pct=best["improvement_pct"],
            max_abs_omega=m["max_abs_omega"],
            max_abs_tau_m=m["max_abs_tau_m"],
            max_abs_u_total=m["max_abs_u_total"],
            final_theta_error=m["final_theta_error"],
            duration=m["duration"],
        )
    )

with open(os.path.join(save_dir, "evaluation_table.json"), "w") as f:
    json.dump(table, f, indent=2)


# 9) Plot training history
plot_training_history(
    history=train_output["history"],
    save_dir=save_dir,
    show=show_plots,
)


# 10) Plot evaluation summary
plot_evaluation_summary(
    rows=rows,
    obj_cfg=obj_cfg,
    save_dir=save_dir,
    show=show_plots,
)


# 11) Plot selected case comparisons
if rows:
    selected_indices = sorted(set([0, len(rows) // 2, len(rows) - 1]))

    for j in selected_indices:
        plot_case_comparison(
            row=rows[j],
            obj_cfg=obj_cfg,
            save_dir=save_dir,
            show=show_plots,
            name_prefix=f"case_{j:02d}",
        )


# 12) Collect final results in one dictionary
results = dict(
    agent=agent,
    train_output=train_output,
    eval_rows=rows,
    train_cases=train_cases,
    test_cases=test_cases,
    traj_cfg=traj_cfg,
    obj_cfg=obj_cfg,
    sac_cfg=sac_cfg,
    save_dir=save_dir,
)

print("\nDone. Results are stored in variable: results")
print(f"Saved outputs to: {save_dir}")
