# -*- coding: utf-8 -*-
"""
true_gps_lqr_guided_addon.py

Clean LQR-guided GPS trajectory teacher/policy file.

This file does not define plant/controller/LQR parameters.  It calls
LQR_TrjOPt_TDESMCwithRLresidual.py for:
    - system parameters
    - base LQR trajectory
    - TDE+SMC rollout and control command generation

Saved files are unchanged:
    save_dir/true_gps_policy.pt
    save_dir/training_teachers.json
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import os
import math
import json
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as exc:  # pragma: no cover
    torch = None
    nn = None
    optim = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

import LQR_TrjOPt_TDESMCwithRLresidual as sysmod


#%% ========================= CONFIG CONTAINERS =========================

@dataclass
class GPSObjectiveConfig:
    omega_limit: float
    torque_limit: float
    command_limit: float
    w_energy: float
    w_theta_track: float
    w_omega_track: float
    w_final_theta: float
    w_final_omega: float
    w_ref_omega_violation: float
    w_actual_omega_violation: float
    w_tau_violation: float
    w_command_violation: float
    w_guide_theta: float
    w_guide_omega: float
    w_duration: float


@dataclass
class GPSTrajectoryConfig:
    n_basis: int
    omega_ref_limit: float
    duration_margin: float
    max_duration_factor: float
    min_extra_duration: float
    duration_exp_scale: float
    z_grid_size: int
    hidden_sizes: Tuple[int, int]


@dataclass
class GPSCase:
    theta_goal: float
    alpha: float
    phi: float

    @property
    def theta_goal_deg(self) -> float:
        return math.degrees(self.theta_goal)

    @property
    def alpha_deg(self) -> float:
        return math.degrees(self.alpha)

    @property
    def phi_deg(self) -> float:
        return math.degrees(self.phi)

    def label(self) -> str:
        return f"goal={self.theta_goal_deg:.1f}deg, alpha={self.alpha_deg:.1f}, phi={self.phi_deg:.1f}"


def mkdir(path: Optional[str]):
    if path:
        os.makedirs(path, exist_ok=True)


def make_default_configs() -> Tuple[GPSTrajectoryConfig, GPSObjectiveConfig]:
    plant_p, _, _, _, _ = sysmod.build_system_from_settings()
    traj_cfg = GPSTrajectoryConfig(
        n_basis=int(sysmod.RL_N_BASIS),
        omega_ref_limit=float(sysmod.RL_OMEGA_REF_LIMIT),
        duration_margin=float(sysmod.RL_DURATION_MARGIN),
        max_duration_factor=float(sysmod.RL_MAX_DURATION_FACTOR),
        min_extra_duration=float(sysmod.RL_MIN_EXTRA_DURATION),
        duration_exp_scale=float(sysmod.RL_DURATION_EXP_SCALE),
        z_grid_size=int(sysmod.RL_Z_GRID_SIZE),
        hidden_sizes=tuple(sysmod.RL_HIDDEN_SIZES),
    )
    obj_cfg = GPSObjectiveConfig(
        omega_limit=float(sysmod.RL_OMEGA_REF_LIMIT),
        torque_limit=2000.0,
        command_limit=float(plant_p.u_max),
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
    return traj_cfg, obj_cfg


def system_for_case(case: GPSCase):
    plant_p, nom, lqr_w, smc_cfg, cost_cfg = sysmod.build_system_from_settings()
    plant_p.alpha = float(case.alpha)
    plant_p.phi = float(case.phi)
    return plant_p, nom, lqr_w, smc_cfg, cost_cfg


def make_cases(goal_degs: Sequence[float], tilt_degs: Sequence[float], coupled_tilts: bool = True) -> List[GPSCase]:
    cases: List[GPSCase] = []
    for g in goal_degs:
        for a in tilt_degs:
            if coupled_tilts:
                cases.append(GPSCase(math.radians(float(g)), math.radians(float(a)), math.radians(float(a))))
            else:
                for p in tilt_degs:
                    cases.append(GPSCase(math.radians(float(g)), math.radians(float(a)), math.radians(float(p))))
    return cases


#%% ========================= LQR-GUIDED RESIDUAL TRAJECTORY =========================

def trapz_safe(y: np.ndarray, x: np.ndarray) -> float:
    return sysmod.trapz_safe(y, x)


def cumtrapz_safe(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return sysmod.cumtrapz_safe(y, x)


def residual_basis(z: np.ndarray, n_basis: int) -> np.ndarray:
    return sysmod.sin_basis(z, n_basis)


def generate_existing_lqr_reference(nom, plant_p, lqr_w, theta0: float, theta_goal: float, dt: float) -> Tuple[np.ndarray, float]:
    """Wrapper around the LQR main file's base-trajectory function."""
    return sysmod.explicit_lqr_reference(
        theta0=theta0,
        theta_goal=theta_goal,
        plant_p=plant_p,
        nom=nom,
        lqr_w=lqr_w,
        duration=getattr(sysmod, "LQR_DURATION_S", None),
    )


def lqr_guide_shape(theta_ref: np.ndarray, theta_goal: float, n_grid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return sysmod.lqr_guide_shape(theta_ref, theta_goal, n_grid)


def decode_duration(raw_duration: float, theta_goal: float, T_guide: float, cfg: GPSTrajectoryConfig) -> float:
    delta = abs(theta_goal)
    if delta < 1e-12:
        return max(0.5, T_guide)
    T_min = cfg.duration_margin * delta / cfg.omega_ref_limit
    T_center = max(T_guide, T_min)
    T_max = max(cfg.max_duration_factor * T_min, T_min + cfg.min_extra_duration, T_center)
    return float(np.clip(T_center * math.exp(cfg.duration_exp_scale * float(raw_duration)), T_min, T_max))


def build_guided_residual_reference(
    theta_goal: float,
    params: np.ndarray,
    base_theta_ref: np.ndarray,
    T_base: float,
    dt: float,
    cfg: GPSTrajectoryConfig,
    theta0: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Dict[str, np.ndarray]]:
    params = np.asarray(params, dtype=float).reshape(-1)
    if params.size != cfg.n_basis + 1:
        raise ValueError(f"params length must be {cfg.n_basis + 1}, got {params.size}")
    delta = float(theta_goal - theta0)
    if abs(delta) < 1e-12:
        T = max(0.5, T_base)
        N = max(1, int(math.ceil(T / dt)))
        t = np.arange(N + 1, dtype=float) * dt
        theta_ref = np.full(N + 1, theta0, dtype=float)
        omega_ref = np.zeros(N + 1, dtype=float)
        extra = dict(z=np.linspace(0.0, 1.0, cfg.z_grid_size), h_base=np.zeros(cfg.z_grid_size), h_gps=np.zeros(cfg.z_grid_size), v_base=np.ones(cfg.z_grid_size), v_gps=np.ones(cfg.z_grid_size))
        return t, theta_ref, omega_ref, float(t[-1]), extra

    z, h_base, v_base = lqr_guide_shape(base_theta_ref, theta_goal, cfg.z_grid_size)
    residual_log = np.clip(residual_basis(z, cfg.n_basis) @ params[:cfg.n_basis], -3.0, 3.0)
    v_gps = np.maximum(v_base, 1e-8) * np.exp(residual_log)
    v_gps = np.maximum(v_gps, 1e-9)
    v_gps /= max(trapz_safe(v_gps, z), 1e-12)

    T = decode_duration(params[-1], theta_goal=theta_goal, T_guide=T_base, cfg=cfg)
    T = max(T, cfg.duration_margin * abs(delta) * float(np.max(v_gps)) / cfg.omega_ref_limit)
    N = max(1, int(math.ceil(T / dt)))
    T = N * dt

    h_gps = cumtrapz_safe(v_gps, z)
    h_gps = h_gps / max(h_gps[-1], 1e-12)
    h_gps = np.maximum.accumulate(np.clip(h_gps, 0.0, 1.0))
    h_gps[0] = 0.0
    h_gps[-1] = 1.0

    t = np.arange(N + 1, dtype=float) * dt
    theta_ref = theta0 + delta * np.interp(t / max(T, 1e-12), z, h_gps)
    theta_ref[0] = theta0
    theta_ref[-1] = theta_goal
    omega_ref = sysmod.finite_diff(theta_ref, dt)
    extra = dict(z=z, h_base=h_base, h_gps=h_gps, v_base=v_base, v_gps=v_gps)
    return t, theta_ref, omega_ref, T, extra


#%% ========================= ROLLOUT / OBJECTIVE VIA LQR MAIN FILE =========================

def simulate_reference(case: GPSCase, theta_ref: np.ndarray, T: float, seed: int = 0, theta0: float = 0.0):
    plant_p, nom, lqr_w, smc_cfg, cost_cfg = system_for_case(case)
    plant = sysmod.OneDOFRotorPlant(plant_p)
    task = sysmod.Task(theta0=theta0, omega0=0.0, theta_goal=case.theta_goal)
    metrics, logs = sysmod.rollout_once(
        plant=plant,
        nom=nom,
        task=task,
        lqr_w=lqr_w,
        smc_cfg=smc_cfg,
        cost_cfg=cost_cfg,
        agent=None,
        reference={"kind": "GPS", "theta": np.asarray(theta_ref, dtype=float), "duration": float(T)},
        seed=seed,
        collect_logs=True,
    )
    return metrics, logs


def compute_objective(logs: Dict[str, np.ndarray], case: GPSCase, obj: GPSObjectiveConfig, guide_theta_ref: Optional[np.ndarray] = None, guide_T: Optional[float] = None) -> Dict[str, float]:
    t = np.asarray(logs["t"], dtype=float)
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else float(sysmod.DT)
    theta = np.asarray(logs["theta"], dtype=float)
    theta_ref = np.asarray(logs["theta_ref"], dtype=float)
    omega = np.asarray(logs["omega"], dtype=float)
    omega_ref = np.asarray(logs["omega_ref"], dtype=float)
    u_total = np.asarray(logs["u_total"], dtype=float)
    tau_m = np.asarray(logs.get("tau_m", u_total), dtype=float)

    goal_scale = max(abs(case.theta_goal), 0.1)
    omega_scale = max(obj.omega_limit, 0.1)
    torque_scale = max(obj.torque_limit, 1.0)
    cmd_scale = max(obj.command_limit, 1.0)

    energy = float(np.sum((u_total / cmd_scale) ** 2) * dt)
    theta_track = float(np.sum(((theta - theta_ref) / goal_scale) ** 2) * dt)
    omega_track = float(np.sum(((omega - omega_ref) / omega_scale) ** 2) * dt)
    final_theta_error = float(theta[-1] - case.theta_goal)
    final_omega = float(omega[-1])

    ref_omega_violation_cost = float(np.sum((np.maximum(0.0, np.abs(omega_ref) - obj.omega_limit) / omega_scale) ** 2) * dt)
    actual_omega_violation_cost = float(np.sum((np.maximum(0.0, np.abs(omega) - obj.omega_limit) / omega_scale) ** 2) * dt)
    tau_violation_cost = float(np.sum((np.maximum(0.0, np.abs(tau_m) - obj.torque_limit) / torque_scale) ** 2) * dt)
    command_violation_cost = float(np.sum((np.maximum(0.0, np.abs(u_total) - obj.command_limit) / cmd_scale) ** 2) * dt)

    guide_theta_cost = 0.0
    guide_omega_cost = 0.0
    if guide_theta_ref is not None:
        z_c = np.linspace(0.0, 1.0, len(theta_ref))
        z_g = np.linspace(0.0, 1.0, len(guide_theta_ref))
        guide_resampled = np.interp(z_c, z_g, np.asarray(guide_theta_ref, dtype=float))
        guide_omega = sysmod.finite_diff(guide_resampled, dt)
        guide_theta_cost = float(np.sum(((theta_ref - guide_resampled) / goal_scale) ** 2) * dt)
        guide_omega_cost = float(np.sum(((omega_ref - guide_omega) / omega_scale) ** 2) * dt)

    duration = float(t[-1]) if len(t) else 0.0
    total = (
        obj.w_energy * energy
        + obj.w_theta_track * theta_track
        + obj.w_omega_track * omega_track
        + obj.w_final_theta * (final_theta_error / goal_scale) ** 2
        + obj.w_final_omega * (final_omega / omega_scale) ** 2
        + obj.w_ref_omega_violation * ref_omega_violation_cost
        + obj.w_actual_omega_violation * actual_omega_violation_cost
        + obj.w_tau_violation * tau_violation_cost
        + obj.w_command_violation * command_violation_cost
        + obj.w_guide_theta * guide_theta_cost
        + obj.w_guide_omega * guide_omega_cost
        + obj.w_duration * duration
    )
    return dict(
        total=float(total), energy=energy, theta_track=theta_track, omega_track=omega_track,
        final_theta_error=abs(final_theta_error), final_omega_abs=abs(final_omega),
        max_abs_omega=float(np.max(np.abs(omega))) if len(omega) else 0.0,
        max_abs_omega_ref=float(np.max(np.abs(omega_ref))) if len(omega_ref) else 0.0,
        max_abs_tau_m=float(np.max(np.abs(tau_m))) if len(tau_m) else 0.0,
        max_abs_u_total=float(np.max(np.abs(u_total))) if len(u_total) else 0.0,
        ref_omega_violation=ref_omega_violation_cost, actual_omega_violation=actual_omega_violation_cost,
        tau_violation=tau_violation_cost, command_violation=command_violation_cost,
        guide_theta=guide_theta_cost, guide_omega=guide_omega_cost, duration=duration,
    )


def evaluate_existing_case(case: GPSCase, obj: GPSObjectiveConfig, seed: int = 0):
    plant_p, nom, lqr_w, _, _ = system_for_case(case)
    theta_ref, T = generate_existing_lqr_reference(nom, plant_p, lqr_w, 0.0, case.theta_goal, plant_p.dt)
    _, logs = simulate_reference(case, theta_ref, T, seed=seed)
    metrics = compute_objective(logs, case, obj)
    return metrics, logs, theta_ref, T


def evaluate_guided_params(case: GPSCase, params: np.ndarray, traj_cfg: GPSTrajectoryConfig, obj_cfg: GPSObjectiveConfig, seed: int = 0):
    plant_p, nom, lqr_w, _, _ = system_for_case(case)
    base_ref, base_T = generate_existing_lqr_reference(nom, plant_p, lqr_w, 0.0, case.theta_goal, plant_p.dt)
    _, theta_ref, _, T, extra = build_guided_residual_reference(case.theta_goal, params, base_ref, base_T, plant_p.dt, traj_cfg, theta0=0.0)
    _, logs = simulate_reference(case, theta_ref, T, seed=seed)
    metrics = compute_objective(logs, case, obj_cfg, guide_theta_ref=base_ref, guide_T=base_T)
    return metrics, logs, theta_ref, T, extra, base_ref, base_T


#%% ========================= CEM LOCAL TEACHER =========================

def cem_optimize_guided_case(
    case: GPSCase,
    traj_cfg: GPSTrajectoryConfig,
    obj_cfg: GPSObjectiveConfig,
    init_mean: Optional[np.ndarray] = None,
    init_std: Optional[np.ndarray] = None,
    cem_iters: int = 10,
    population: int = 48,
    elite_frac: float = 0.20,
    seed: int = 0,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    dim = traj_cfg.n_basis + 1
    mean = np.zeros(dim, dtype=float) if init_mean is None else np.asarray(init_mean, dtype=float).copy()
    std = np.ones(dim, dtype=float) * 0.65 if init_std is None else np.asarray(init_std, dtype=float).copy()
    std[-1] = min(std[-1], 0.45)
    elite_n = max(2, int(round(population * elite_frac)))
    best = dict(cost=np.inf, params=None, metrics=None, logs=None, theta_ref=None, T=None, extra=None)
    history: List[Dict[str, float]] = []

    for it in range(cem_iters):
        candidates = rng.normal(mean, std, size=(population, dim))
        candidates[0] = mean
        if population > 1:
            candidates[1] = np.zeros(dim)
        scored = []
        for j, p in enumerate(candidates):
            try:
                metrics, logs, theta_ref, T, extra, _, _ = evaluate_guided_params(case, p, traj_cfg, obj_cfg, seed=seed + 1000 * it + j)
                cost = metrics["total"]
            except Exception:
                metrics, logs, theta_ref, T, extra = None, None, None, None, None
                cost = np.inf
            scored.append((cost, p, metrics, logs, theta_ref, T, extra))
        scored.sort(key=lambda x: x[0])
        elites = np.stack([s[1] for s in scored[:elite_n]], axis=0)
        mean = elites.mean(axis=0)
        std = np.minimum(elites.std(axis=0) + 1e-4, 1.25)
        std[-1] = min(std[-1], 0.60)
        if scored[0][0] < best["cost"]:
            cost, p, metrics, logs, theta_ref, T, extra = scored[0]
            best.update(cost=float(cost), params=p.copy(), metrics=metrics, logs=logs, theta_ref=theta_ref, T=T, extra=extra)
        history.append(dict(iteration=it, best=float(scored[0][0]), mean=float(np.mean([s[0] for s in scored if np.isfinite(s[0])]))))
        print(f"CEM {case.label()} iter {it+1:02d}/{cem_iters}: best={scored[0][0]:.6g}")
    return dict(case=case, best=best, final_mean=mean, final_std=std, history=history)


#%% ========================= GLOBAL POLICY =========================

class GuidedTrajectoryPolicy(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Tuple[int, int]):
        if torch is None:
            raise RuntimeError(f"PyTorch import failed: {_TORCH_IMPORT_ERROR}")
        super().__init__()
        layers: List[nn.Module] = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h)); layers.append(nn.Tanh()); last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def case_features(case: GPSCase) -> np.ndarray:
    max_tilt = math.radians(float(sysmod.RL_MAX_TILT_DEG))
    return np.array([case.theta_goal / math.pi, case.alpha / max_tilt, case.phi / max_tilt], dtype=np.float32)


def train_policy_from_local_teachers(local_results: Sequence[Dict[str, object]], traj_cfg: GPSTrajectoryConfig, epochs: int = 1500, lr: float = 1e-3, seed: int = 0):
    if torch is None:
        raise RuntimeError(f"PyTorch import failed: {_TORCH_IMPORT_ERROR}")
    torch.manual_seed(seed); np.random.seed(seed)
    X, Y = [], []
    for res in local_results:
        X.append(case_features(res["case"]))
        Y.append(np.asarray(res["best"]["params"], dtype=np.float32))
    X_t = torch.as_tensor(np.stack(X, axis=0), dtype=torch.float32)
    Y_t = torch.as_tensor(np.stack(Y, axis=0), dtype=torch.float32)
    policy = GuidedTrajectoryPolicy(3, traj_cfg.n_basis + 1, traj_cfg.hidden_sizes)
    opt = optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-5)
    losses = []
    for ep in range(int(epochs)):
        pred = policy(X_t)
        loss = torch.mean((pred - Y_t) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach().cpu().item()))
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"policy imitation epoch {ep+1}/{epochs}: loss={losses[-1]:.6g}")
    return policy, losses


def policy_predict_params(policy: GuidedTrajectoryPolicy, case: GPSCase) -> np.ndarray:
    x = torch.as_tensor(case_features(case), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return policy(x).squeeze(0).cpu().numpy().astype(float)


def train_true_gps_policy(train_cases: Sequence[GPSCase], traj_cfg: GPSTrajectoryConfig, obj_cfg: GPSObjectiveConfig, cem_iters: int = 10, population: int = 48, elite_frac: float = 0.20, policy_epochs: int = 1500, seed: int = 0, save_dir: Optional[str] = None):
    mkdir(save_dir)
    local_results = []
    for i, case in enumerate(train_cases):
        print(f"\n=== Local GPS teacher {i+1}/{len(train_cases)}: {case.label()} ===")
        res = cem_optimize_guided_case(case, traj_cfg, obj_cfg, cem_iters=cem_iters, population=population, elite_frac=elite_frac, seed=seed + 100 * i)
        local_results.append(res)
    policy, losses = train_policy_from_local_teachers(local_results, traj_cfg, epochs=policy_epochs, seed=seed)
    if save_dir:
        # Keep saved model names/formats unchanged.
        torch.save(policy.state_dict(), os.path.join(save_dir, "true_gps_policy.pt"))
        with open(os.path.join(save_dir, "training_teachers.json"), "w") as f:
            json.dump([dict(theta_goal_deg=r["case"].theta_goal_deg, alpha_deg=r["case"].alpha_deg, phi_deg=r["case"].phi_deg, best_cost=r["best"]["cost"], best_params=np.asarray(r["best"]["params"]).tolist()) for r in local_results], f, indent=2)
    return dict(policy=policy, losses=losses, local_results=local_results)


def evaluate_true_gps_policy(policy: GuidedTrajectoryPolicy, test_cases: Sequence[GPSCase], traj_cfg: GPSTrajectoryConfig, obj_cfg: GPSObjectiveConfig, refinement_iters: int = 4, refinement_population: int = 24, seed: int = 1000):
    rows = []
    for i, case in enumerate(test_cases):
        print(f"\n=== Evaluate {i+1}/{len(test_cases)}: {case.label()} ===")
        existing_metrics, existing_logs, existing_ref, existing_T = evaluate_existing_case(case, obj_cfg, seed=seed + i)
        p_nn = policy_predict_params(policy, case)
        nn_metrics, nn_logs, nn_ref, nn_T, nn_extra, base_ref, base_T = evaluate_guided_params(case, p_nn, traj_cfg, obj_cfg, seed=seed + i)
        ref_result = cem_optimize_guided_case(case, traj_cfg, obj_cfg, init_mean=p_nn, init_std=np.r_[np.ones(traj_cfg.n_basis) * 0.25, 0.20], cem_iters=refinement_iters, population=refinement_population, elite_frac=0.25, seed=seed + 5000 + i)
        rows.append(dict(case=case, existing_metrics=existing_metrics, existing_logs=existing_logs, existing_ref=existing_ref, existing_T=existing_T, nn_params=p_nn, nn_metrics=nn_metrics, nn_logs=nn_logs, nn_ref=nn_ref, nn_T=nn_T, nn_extra=nn_extra, gps_params=ref_result["best"]["params"], gps_metrics=ref_result["best"]["metrics"], gps_logs=ref_result["best"]["logs"], gps_ref=ref_result["best"]["theta_ref"], gps_T=ref_result["best"]["T"], gps_extra=ref_result["best"]["extra"], base_ref=base_ref, base_T=base_T, refinement_history=ref_result["history"]))
    return rows


#%% ========================= PLOTTING / RUNNER =========================

def _save_or_show(fig, save_dir: Optional[str], name: str, show: bool):
    if save_dir:
        mkdir(save_dir); fig.savefig(os.path.join(save_dir, name), dpi=180, bbox_inches="tight")
    if show: plt.show()
    else: plt.close(fig)


def plot_training_diagnostics(train_output: Dict[str, object], save_dir: Optional[str] = None, show: bool = True):
    losses = train_output["losses"]
    fig = plt.figure(figsize=(8, 4.5)); plt.plot(losses); plt.yscale("log")
    plt.xlabel("policy imitation epoch"); plt.ylabel("MSE loss"); plt.title("Deep GPS policy learns LQR-guided teachers")
    plt.grid(True, linestyle="--", alpha=0.6); _save_or_show(fig, save_dir, "policy_imitation_loss.png", show)


def plot_evaluation_summary(rows: Sequence[Dict[str, object]], save_dir: Optional[str] = None, show: bool = True):
    if not rows: return
    labels = [f"{r['case'].theta_goal_deg:.0f}\n{r['case'].alpha_deg:.0f}/{r['case'].phi_deg:.0f}" for r in rows]
    x = np.arange(len(rows)); width = 0.25
    c_existing = np.array([r["existing_metrics"]["total"] for r in rows])
    c_nn = np.array([r["nn_metrics"]["total"] for r in rows])
    c_gps = np.array([r["gps_metrics"]["total"] for r in rows])
    fig = plt.figure(figsize=(max(10, len(rows) * 0.55), 5))
    plt.bar(x - width, c_existing, width, label="LQR")
    plt.bar(x, c_nn, width, label="policy")
    plt.bar(x + width, c_gps, width, label="GPS refined")
    plt.xticks(x, labels, rotation=45, ha="right"); plt.ylabel("objective cost"); plt.title("Trajectory objective comparison")
    plt.legend(); plt.grid(True, axis="y", linestyle="--", alpha=0.6); _save_or_show(fig, save_dir, "summary_objective_cost.png", show)


def save_evaluation_table(rows: Sequence[Dict[str, object]], save_dir: str):
    mkdir(save_dir)
    table = []
    for r in rows:
        c = r["case"]
        table.append(dict(theta_goal_deg=c.theta_goal_deg, alpha_deg=c.alpha_deg, phi_deg=c.phi_deg, existing_cost=r["existing_metrics"]["total"], nn_cost=r["nn_metrics"]["total"], gps_cost=r["gps_metrics"]["total"]))
    with open(os.path.join(save_dir, "true_gps_evaluation_summary.json"), "w") as f:
        json.dump(table, f, indent=2)


def run_true_gps_experiment(
    train_goal_degs: Sequence[float] = (0, 30, 60, 90, 120, 150, 180),
    test_goal_degs: Sequence[float] = (15, 45, 75, 105, 135, 165, 180),
    tilt_degs: Sequence[float] = (0, 10, 20),
    coupled_tilts: bool = True,
    cem_iters: int = 8,
    population: int = 36,
    policy_epochs: int = 1200,
    refinement_iters: int = 4,
    refinement_population: int = 24,
    save_dir: str = "true_gps_results",
    show_plots: bool = True,
):
    mkdir(save_dir)
    traj_cfg, obj_cfg = make_default_configs()
    train_cases = make_cases(train_goal_degs, tilt_degs, coupled_tilts=coupled_tilts)
    train_output = train_true_gps_policy(train_cases, traj_cfg, obj_cfg, cem_iters=cem_iters, population=population, elite_frac=0.20, policy_epochs=policy_epochs, seed=1, save_dir=save_dir)
    test_cases = make_cases(test_goal_degs, tilt_degs, coupled_tilts=coupled_tilts)
    rows = evaluate_true_gps_policy(train_output["policy"], test_cases, traj_cfg, obj_cfg, refinement_iters=refinement_iters, refinement_population=refinement_population, seed=1000)
    plot_training_diagnostics(train_output, save_dir=save_dir, show=show_plots)
    plot_evaluation_summary(rows, save_dir=save_dir, show=show_plots)
    save_evaluation_table(rows, save_dir)
    print(f"\nResults saved in: {os.path.abspath(save_dir)}")
    return dict(train_output=train_output, evaluation_rows=rows, traj_cfg=traj_cfg, obj_cfg=obj_cfg)


#%% ========================= RUN IN SPYDER =========================
if __name__ == "__main__":
    results = run_true_gps_experiment(
        train_goal_degs=(30, 60, 90, 120, 150, 180),
        test_goal_degs=(15, 45, 75, 105, 135, 165, 180),
        tilt_degs=(0,5,8,15, 20),
        cem_iters=30,
        population=50,
        policy_epochs=3000,
        refinement_iters=2,
        refinement_population=10,
        save_dir="true_gps_results_smoke",
        show_plots=True,
    )
