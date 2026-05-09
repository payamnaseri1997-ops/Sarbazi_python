# -*- coding: utf-8 -*-
"""
trajectory_rl_gps_addon.py

Clean NN/GPS trajectory-learning add-on.

This file trains a neural trajectory generator, but it does NOT define its own
plant, controller, LQR model, or physical parameters.  Every system rollout,
base LQR trajectory, and TDE+SMC command comes from
LQR_TrjOPt_TDESMCwithRLresidual.py.

Cost terminology:
    - The base LQR trajectory uses the fixed-horizon nominal LQR planning
      objective in the main file.  Its duration is prescribed by
      LQR_DURATION_S, or by time_horizon(...) when LQR_DURATION_S is None;
      LQR does not optimize duration and has no w_time/J_time term.
    - The NN/GPS trajectory objective below is a separate closed-loop rollout
      objective.  It may include duration because this learned trajectory
      parameterization chooses duration.

Saved model format is unchanged:
    save_dir/trajectory_policy.pt
    save_dir/training_cases.json
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import os
import math
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

import LQR_TrjOPt_TDESMCwithRLresidual as sysmod


#%% ========================= CONFIG CONTAINERS =========================

@dataclass
class TrajectoryConstraintConfig:
    omega_limit: float
    torque_limit: float
    u_command_limit: float
    w_energy: float
    w_track: float
    w_vel_track: float
    w_final: float
    w_ref_vel_violation: float
    w_actual_vel_violation: float
    w_torque_violation: float
    w_command_violation: float
    w_duration: float


@dataclass
class TrajectoryPolicyConfig:
    n_shape: int
    omega_ref_limit: float
    min_duration: float
    max_duration: float
    hidden_sizes: Tuple[int, int]


@dataclass
class CaseSpec:
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


def make_default_configs() -> Tuple[TrajectoryPolicyConfig, TrajectoryConstraintConfig]:
    """Read trajectory/system values from the LQR main file."""
    plant_p, _, _, _, _ = sysmod.build_system_from_settings()
    traj_cfg = TrajectoryPolicyConfig(
        n_shape=int(sysmod.NN_N_SHAPE),
        omega_ref_limit=float(sysmod.NN_OMEGA_REF_LIMIT),
        min_duration=float(sysmod.NN_MIN_DURATION),
        max_duration=float(sysmod.NN_MAX_DURATION),
        hidden_sizes=tuple(sysmod.NN_HIDDEN_SIZES),
    )
    obj_cfg = TrajectoryConstraintConfig(
        omega_limit=float(sysmod.NN_OMEGA_REF_LIMIT),
        torque_limit=2000.0,
        u_command_limit=float(plant_p.u_max),
        w_energy=1.0,
        w_track=10.0,
        w_vel_track=2.0,
        w_final=500.0,
        w_ref_vel_violation=1.0e4,
        w_actual_vel_violation=1.0e4,
        w_torque_violation=1.0e4,
        w_command_violation=1.0e4,
        w_duration=0.0,
    )
    return traj_cfg, obj_cfg


def system_for_case(case: CaseSpec):
    """Use exactly the LQR main-file settings, overriding only case tilt."""
    plant_p, nom, lqr_w, smc_cfg, cost_cfg = sysmod.build_system_from_settings()
    plant_p.alpha = float(case.alpha)
    plant_p.phi = float(case.phi)
    return plant_p, nom, lqr_w, smc_cfg, cost_cfg


def make_cases(goal_degs: Sequence[float], tilt_degs: Sequence[float], coupled_tilt: bool = True) -> List[CaseSpec]:
    cases: List[CaseSpec] = []
    for g in goal_degs:
        for a in tilt_degs:
            if coupled_tilt:
                cases.append(CaseSpec(math.radians(float(g)), math.radians(float(a)), math.radians(float(a))))
            else:
                for p in tilt_degs:
                    cases.append(CaseSpec(math.radians(float(g)), math.radians(float(a)), math.radians(float(p))))
    return cases


# ========================= TRAJECTORY PARAMETERIZATION =========================

def _softplus_np(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _shape_basis(z: np.ndarray, n_shape: int) -> np.ndarray:
    return sysmod.sin_basis(z, n_shape)



HOLD_FRACTION_AFTER_GOAL = 0.25


def _extend_reference_with_goal_hold(
    theta_ref: np.ndarray,
    dt: float,
    hold_fraction: float = HOLD_FRACTION_AFTER_GOAL,
    theta_goal: Optional[float] = None,
) -> np.ndarray:
    """Append a flat tail at theta_goal for a fraction of current trajectory time."""
    theta_ref = np.asarray(theta_ref, dtype=float).reshape(-1)
    if theta_ref.size < 2 or hold_fraction <= 0.0:
        return theta_ref
    hold_steps = int(round((theta_ref.size - 1) * hold_fraction))
    if hold_steps <= 0:
        return theta_ref
    goal_value = float(theta_ref[-1] if theta_goal is None else theta_goal)
    return np.r_[theta_ref, np.full(hold_steps, goal_value, dtype=float)]


def build_monotone_reference(
    theta_goal: float,
    params: np.ndarray,
    dt: float,
    cfg: TrajectoryPolicyConfig,
    theta0: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Same saved-policy decoder shape; uses sysmod finite_diff/trapz helpers."""
    params = np.asarray(params, dtype=float).reshape(-1)
    if params.size != cfg.n_shape + 1:
        raise ValueError(f"params must have length {cfg.n_shape + 1}, got {params.size}")

    delta = float(theta_goal - theta0)
    if abs(delta) < 1e-12:
        t = np.arange(0.0, cfg.min_duration + 0.5 * dt, dt)
        theta_ref = np.full_like(t, theta0, dtype=float)
        omega_ref = np.zeros_like(t)
        return t, theta_ref, omega_ref, float(t[-1])

    z_grid = np.linspace(0.0, 1.0, 1001)
    logits = _shape_basis(z_grid, cfg.n_shape) @ params[:cfg.n_shape]
    v_shape = z_grid * (1.0 - z_grid) * np.exp(np.clip(logits, -5.0, 5.0)) + 1e-6
    v_norm = v_shape / max(sysmod.trapz_safe(v_shape, z_grid), 1e-12)
    h_grid = sysmod.cumtrapz_safe(v_norm, z_grid)
    h_grid /= max(h_grid[-1], 1e-12)

    max_hprime = float(np.max(np.abs(v_norm)))
    T_min_vel = abs(delta) * max_hprime / max(cfg.omega_ref_limit, 1e-12)
    duration_slack = float(_softplus_np(np.array([params[-1]]))[0])
    T = float(np.clip(T_min_vel + duration_slack, cfg.min_duration, cfg.max_duration))

    N = max(1, int(round(T / dt)))
    t = np.arange(N + 1, dtype=float) * dt
    T = float(t[-1])
    h = np.interp(np.clip(t / max(T, 1e-12), 0.0, 1.0), z_grid, h_grid)
    theta_ref = theta0 + delta * h
    theta_ref[0] = theta0
    theta_ref[-1] = theta_goal
    theta_ref = _extend_reference_with_goal_hold(theta_ref, dt, theta_goal=theta_goal)
    omega_ref = sysmod.finite_diff(theta_ref, dt)
    t = np.arange(theta_ref.size, dtype=float) * dt
    T = float(t[-1])
    return t, theta_ref, omega_ref, T


# ========================= SYSTEM ROLLOUT THROUGH LQR MAIN FILE =========================

def base_lqr_reference(case: CaseSpec, theta0: float = 0.0) -> Tuple[np.ndarray, float]:
    # Fixed-horizon nominal LQR reference from the main file.  LQR_DURATION_S
    # prescribes the horizon when set; otherwise time_horizon(...) is used.
    # The LQR planner does not optimize duration.
    plant_p, nom, lqr_w, _, _ = system_for_case(case)
    return sysmod.explicit_lqr_reference(
        theta0=theta0,
        theta_goal=case.theta_goal,
        plant_p=plant_p,
        nom=nom,
        lqr_w=lqr_w,
        duration=getattr(sysmod, "LQR_DURATION_S", None),
    )


def simulate_reference(case: CaseSpec, theta_ref: np.ndarray, duration: float, seed: int = 0, theta0: float = 0.0):
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
        reference={"theta": np.asarray(theta_ref, dtype=float), "duration": float(duration), "kind": "NN-GPS"},
        seed=seed,
        collect_logs=True,
    )
    return metrics, logs, plant_p


def simulate_existing_reference(case: CaseSpec, seed: int = 0, theta0: float = 0.0):
    theta_ref, T = base_lqr_reference(case, theta0=theta0)
    metrics, logs, plant_p = simulate_reference(case, theta_ref, T, seed=seed, theta0=theta0)
    return metrics, logs, theta_ref, T, plant_p


def trajectory_objective(logs: Dict[str, np.ndarray], theta_goal: float, plant_p, obj_cfg: TrajectoryConstraintConfig) -> Dict[str, float]:
    # Closed-loop trajectory-learning objective evaluated after rollout.
    # This is separate from the finite-horizon LQR planning objective used to
    # generate the baseline theta_ref.  Its duration term belongs to NN/GPS
    # teacher evaluation, not to the LQR planner.
    required_keys = ("t", "theta", "theta_ref", "omega", "omega_ref", "u_total")
    missing = [k for k in required_keys if k not in logs]
    if missing:
        raise KeyError(f"trajectory_objective missing required log keys: {missing}")
    t = np.asarray(logs["t"], dtype=float)
    if t.size == 0:
        return {"total_cost": 1e12}
    dt = float(plant_p.dt)
    theta = np.asarray(logs["theta"], dtype=float)
    theta_ref = np.asarray(logs["theta_ref"], dtype=float)
    omega = np.asarray(logs["omega"], dtype=float)
    omega_ref = np.asarray(logs["omega_ref"], dtype=float)
    u_total = np.asarray(logs["u_total"], dtype=float)
    tau_m = np.asarray(logs.get("tau_m", u_total), dtype=float)

    e = theta - theta_ref
    edot = omega - omega_ref
    energy = float(np.sum((u_total / max(obj_cfg.u_command_limit, 1e-12)) ** 2) * dt)
    track = float(np.sum(e ** 2) * dt)
    vel_track = float(np.sum(edot ** 2) * dt)
    final_theta_error = float(theta[-1] - theta_goal)
    final_omega = float(omega[-1])
    final = final_theta_error ** 2 + final_omega ** 2

    ref_vel_violation = float(np.sum(np.maximum(np.abs(omega_ref) / obj_cfg.omega_limit - 1.0, 0.0) ** 2) * dt)
    actual_vel_violation = float(np.sum(np.maximum(np.abs(omega) / obj_cfg.omega_limit - 1.0, 0.0) ** 2) * dt)
    torque_violation = float(np.sum(np.maximum(np.abs(tau_m) / obj_cfg.torque_limit - 1.0, 0.0) ** 2) * dt)
    command_violation = float(np.sum(np.maximum(np.abs(u_total) / obj_cfg.u_command_limit - 1.0, 0.0) ** 2) * dt)
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0

    total = (
        obj_cfg.w_energy * energy
        + obj_cfg.w_track * track
        + obj_cfg.w_vel_track * vel_track
        + obj_cfg.w_final * final
        + obj_cfg.w_ref_vel_violation * ref_vel_violation
        + obj_cfg.w_actual_vel_violation * actual_vel_violation
        + obj_cfg.w_torque_violation * torque_violation
        + obj_cfg.w_command_violation * command_violation
        + obj_cfg.w_duration * duration
    )
    return dict(
        total_cost=float(total), energy=energy, track=track, vel_track=vel_track, final=final,
        final_theta_error=final_theta_error, final_omega=final_omega,
        ref_vel_violation=ref_vel_violation, actual_vel_violation=actual_vel_violation,
        torque_violation=torque_violation, command_violation=command_violation, duration=duration,
        max_abs_omega=float(np.max(np.abs(omega))),
        max_abs_omega_ref=float(np.max(np.abs(omega_ref))),
        max_abs_tau_m=float(np.max(np.abs(tau_m))),
        max_abs_u_total=float(np.max(np.abs(u_total))),
    )


def evaluate_params_for_case(case: CaseSpec, params: np.ndarray, traj_cfg: TrajectoryPolicyConfig, obj_cfg: TrajectoryConstraintConfig, seed: int = 0):
    plant_p, _, _, _, _ = system_for_case(case)
    _, theta_ref, _, T = build_monotone_reference(case.theta_goal, params, plant_p.dt, traj_cfg, theta0=0.0)
    _, logs, plant_p = simulate_reference(case, theta_ref, T, seed=seed)
    metrics = trajectory_objective(logs, case.theta_goal, plant_p, obj_cfg)
    return metrics, logs, theta_ref, T


# ========================= LOCAL CEM TEACHER =========================

def cem_optimize_case(
    case: CaseSpec,
    traj_cfg: TrajectoryPolicyConfig,
    obj_cfg: TrajectoryConstraintConfig,
    n_iter: int = 12,
    population: int = 32,
    elite_frac: float = 0.20,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    dim = traj_cfg.n_shape + 1
    elite_count = max(2, int(round(population * elite_frac)))
    mean = np.zeros(dim, dtype=float)
    std = np.ones(dim, dtype=float)
    std[:traj_cfg.n_shape] = 0.75
    std[-1] = 2.0

    best = dict(cost=np.inf, params=mean.copy(), metrics=None, logs=None, theta_ref=None, duration=None)
    history: List[Dict[str, float]] = []
    base_metrics, base_logs, _, _, _ = simulate_existing_reference(case, seed=seed)

    for it in range(1, n_iter + 1):
        samples = rng.normal(mean, std, size=(population, dim))
        samples[0] = mean
        if population > 1:
            samples[1] = 0.0
        samples[:, :traj_cfg.n_shape] = np.clip(samples[:, :traj_cfg.n_shape], -4.0, 4.0)
        samples[:, -1] = np.clip(samples[:, -1], -4.0, 12.0)

        costs = np.zeros(population, dtype=float)
        eval_cache = []
        for i, p in enumerate(samples):
            metrics, logs, theta_ref, T = evaluate_params_for_case(case, p, traj_cfg, obj_cfg, seed=seed + i)
            costs[i] = metrics["total_cost"]
            eval_cache.append((metrics, logs, theta_ref, T))

        order = np.argsort(costs)
        elites = samples[order[:elite_count]]
        if costs[order[0]] < best["cost"]:
            metrics, logs, theta_ref, T = eval_cache[order[0]]
            best.update(cost=float(costs[order[0]]), params=samples[order[0]].copy(), metrics=metrics, logs=logs, theta_ref=theta_ref, duration=T)

        mean = 0.35 * mean + 0.65 * elites.mean(axis=0)
        std = 0.35 * std + 0.65 * (elites.std(axis=0) + 1e-3)
        std = np.maximum(std, 0.03)
        history.append(dict(iteration=it, best=float(best["cost"]), baseline=float(base_metrics["total_cost"])))
        if verbose:
            print(f"CEM {case.label()} iter={it:02d}: best={best['cost']:.5g}, baseline={base_metrics['total_cost']:.5g}")

    best["history"] = history
    best["baseline_metrics"] = base_metrics
    best["baseline_logs"] = base_logs
    return best


# ========================= DEEP TRAJECTORY POLICY =========================

class TrajectoryPolicyNet:
    def __init__(self, cfg: TrajectoryPolicyConfig):
        h1, h2 = cfg.hidden_sizes
        out = cfg.n_shape + 1
        rng = np.random.default_rng(0)
        self.W1 = rng.normal(0.0, 0.2, size=(3, h1))
        self.b1 = np.zeros(h1, dtype=float)
        self.W2 = rng.normal(0.0, 0.2, size=(h1, h2))
        self.b2 = np.zeros(h2, dtype=float)
        self.W3 = rng.normal(0.0, 0.2, size=(h2, out))
        self.b3 = np.zeros(out, dtype=float)

    def forward(self, x: np.ndarray) -> np.ndarray:
        a1 = np.tanh(x @ self.W1 + self.b1)
        a2 = np.tanh(a1 @ self.W2 + self.b2)
        return a2 @ self.W3 + self.b3

    def state_dict(self) -> Dict[str, np.ndarray]:
        return dict(W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)

    def load_state_dict(self, state: Dict[str, np.ndarray]):
        for k in ("W1", "b1", "W2", "b2", "W3", "b3"):
            setattr(self, k, np.asarray(state[k], dtype=float))


def case_to_policy_input(case: CaseSpec) -> np.ndarray:
    max_tilt = math.radians(float(sysmod.NN_MAX_TILT_DEG))
    return np.array([case.theta_goal / math.pi, case.alpha / max_tilt, case.phi / max_tilt], dtype=np.float32)


def fit_policy_to_local_solutions(policy: TrajectoryPolicyNet, cases: Sequence[CaseSpec], params_list: Sequence[np.ndarray], epochs: int = 1500, lr: float = 2e-3, verbose: bool = True):
    x = np.stack([case_to_policy_input(c) for c in cases]).astype(float)
    y = np.stack([np.asarray(p, dtype=float) for p in params_list]).astype(float)
    losses = []
    for ep in range(1, int(epochs) + 1):
        z1 = x @ policy.W1 + policy.b1
        a1 = np.tanh(z1)
        z2 = a1 @ policy.W2 + policy.b2
        a2 = np.tanh(z2)
        pred = a2 @ policy.W3 + policy.b3
        err = pred - y
        loss = float(np.mean(err ** 2))
        n = x.shape[0]
        dY = (2.0 / (n * err.shape[1])) * err
        dW3 = a2.T @ dY
        db3 = dY.sum(axis=0)
        dA2 = dY @ policy.W3.T
        dZ2 = dA2 * (1.0 - np.tanh(z2) ** 2)
        dW2 = a1.T @ dZ2
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ policy.W2.T
        dZ1 = dA1 * (1.0 - np.tanh(z1) ** 2)
        dW1 = x.T @ dZ1
        db1 = dZ1.sum(axis=0)
        policy.W3 -= lr * dW3; policy.b3 -= lr * db3
        policy.W2 -= lr * dW2; policy.b2 -= lr * db2
        policy.W1 -= lr * dW1; policy.b1 -= lr * db1
        losses.append(loss)
        if verbose and (ep == 1 or ep % 250 == 0 or ep == epochs):
            print(f"policy fit epoch {ep:04d}/{epochs}: mse={losses[-1]:.6g}")
    return losses


def policy_params(policy: TrajectoryPolicyNet, case: CaseSpec) -> np.ndarray:
    x = case_to_policy_input(case).astype(float)[None, :]
    p = policy.forward(x).squeeze(0)
    p[:-1] = np.clip(p[:-1], -4.0, 4.0)
    p[-1] = np.clip(p[-1], -4.0, 12.0)
    return p.astype(float)



def _model_path(save_dir: str, filename: str) -> str:
    return os.path.join(save_dir, filename)


def _case_key(case: CaseSpec) -> str:
    return f"{case.theta_goal_deg:.6f}|{case.alpha_deg:.6f}|{case.phi_deg:.6f}"


def _case_to_dict(case: CaseSpec) -> Dict[str, float]:
    return {"theta_goal_deg": float(case.theta_goal_deg), "alpha_deg": float(case.alpha_deg), "phi_deg": float(case.phi_deg)}


def _load_training_history(save_dir: str) -> List[Dict[str, object]]:
    path = _model_path(save_dir, "training_history.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def _save_training_history(save_dir: str, run_record: Dict[str, object]) -> str:
    path = _model_path(save_dir, "training_history.json")
    history = _load_training_history(save_dir)
    history.append(run_record)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    return path

def train_gps_trajectory_policy(
    train_cases: Sequence[CaseSpec],
    traj_cfg: Optional[TrajectoryPolicyConfig] = None,
    obj_cfg: Optional[TrajectoryConstraintConfig] = None,
    cem_iters: int = 12,
    population: int = 32,
    elite_frac: float = 0.20,
    policy_epochs: int = 1500,
    seed: int = 1,
    save_dir: str = "trajectory_rl_results",
    resume_training: bool = False,
    skip_existing_cases: bool = False,
) -> Dict[str, object]:
    if traj_cfg is None or obj_cfg is None:
        default_traj, default_obj = make_default_configs()
        traj_cfg = default_traj if traj_cfg is None else traj_cfg
        obj_cfg = default_obj if obj_cfg is None else obj_cfg
    os.makedirs(save_dir, exist_ok=True)

    timestamp = str(np.datetime64("now"))
    model_path = _model_path(save_dir, "trajectory_policy.pt")
    teachers_path = _model_path(save_dir, "training_cases.json")
    if resume_training:
        if os.path.exists(model_path):
            print(f"Resuming training from: {model_path}")
        else:
            raise FileNotFoundError(f"resume_training=True but no saved model found at: {model_path}")
    else:
        print("Starting training from scratch")

    prior_case_keys = set()
    if os.path.exists(teachers_path):
        with open(teachers_path, "r") as f:
            for row in json.load(f):
                prior_case_keys.add(f"{float(row['theta_goal_deg']):.6f}|{float(row['alpha_deg']):.6f}|{float(row['phi_deg']):.6f}")

    selected_cases = []
    skipped_cases = []
    for c in train_cases:
        if skip_existing_cases and _case_key(c) in prior_case_keys:
            skipped_cases.append(c)
        else:
            selected_cases.append(c)

    local_results = []
    params_list = []
    for idx, case in enumerate(selected_cases):
        result = cem_optimize_case(case, traj_cfg, obj_cfg, n_iter=cem_iters, population=population, elite_frac=elite_frac, seed=seed + 100 * idx, verbose=True)
        local_results.append(result)
        params_list.append(result["params"])

    policy = TrajectoryPolicyNet(traj_cfg)
    if resume_training:
        with np.load(model_path, allow_pickle=False) as data:
            policy.load_state_dict({k: data[k] for k in data.files})
    losses = fit_policy_to_local_solutions(policy, selected_cases, params_list, epochs=policy_epochs, lr=2e-3) if selected_cases else []

    with open(model_path, "wb") as f:
        np.savez(f, **policy.state_dict())
    print(f"Saved model to: {model_path}")
    with open(teachers_path, "w") as f:
        json.dump([_case_to_dict(c) for c in train_cases], f, indent=2)

    run_record = dict(timestamp=timestamp, resume_training=bool(resume_training), skip_existing_cases=bool(skip_existing_cases), cem_iters=int(cem_iters), population=int(population), policy_epochs=int(policy_epochs), model_path=model_path, trained_cases_this_run=[_case_to_dict(c) for c in selected_cases], skipped_cases=[_case_to_dict(c) for c in skipped_cases])
    history_path = _save_training_history(save_dir, run_record)
    latest_path = _model_path(save_dir, "latest_run_config.json")
    with open(latest_path, "w") as f:
        json.dump(run_record, f, indent=2)
    print(f"Saved training metadata to: {latest_path}")

    return dict(policy=policy, traj_cfg=traj_cfg, obj_cfg=obj_cfg, local_results=local_results, policy_losses=losses, train_cases=list(selected_cases), skipped_cases=skipped_cases, save_dir=save_dir, history_path=history_path)


def evaluate_policy_vs_existing(policy: TrajectoryPolicyNet, cases: Sequence[CaseSpec], traj_cfg: TrajectoryPolicyConfig, obj_cfg: TrajectoryConstraintConfig, seed: int = 123):
    rows = []
    for i, case in enumerate(cases):
        _, base_logs, base_theta_ref, base_T, plant_p = simulate_existing_reference(case, seed=seed + i)
        base_metrics = trajectory_objective(base_logs, case.theta_goal, plant_p, obj_cfg)
        p = policy_params(policy, case)
        rl_metrics, rl_logs, rl_theta_ref, rl_T = evaluate_params_for_case(case, p, traj_cfg, obj_cfg, seed=seed + i)
        rows.append(dict(case=case, existing_metrics=base_metrics, rl_metrics=rl_metrics, existing_logs=base_logs, rl_logs=rl_logs, existing_theta_ref=base_theta_ref, rl_theta_ref=rl_theta_ref, existing_duration=base_T, rl_duration=rl_T, rl_params=p, improvement=base_metrics["total_cost"] - rl_metrics["total_cost"], improvement_ratio=(base_metrics["total_cost"] - rl_metrics["total_cost"]) / max(abs(base_metrics["total_cost"]), 1e-12)))
        print(f"Eval {case.label()}: fixed-horizon LQR rollout objective={base_metrics['total_cost']:.5g}, NN-traj rollout objective={rl_metrics['total_cost']:.5g}, improvement={rows[-1]['improvement_ratio'] * 100:.1f}%")
    return rows


# ========================= PLOTS / RUNNER =========================

def _maybe_save(fig, save_dir: Optional[str], name: str, show: bool):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, name), dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_training_progress(train_output: Dict[str, object], save_dir: Optional[str] = None, show: bool = True):
    losses = np.asarray(train_output["policy_losses"], dtype=float)
    fig = plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(losses) + 1), losses)
    plt.xlabel("Supervised GPS policy-fit epoch")
    plt.ylabel("MSE to local optimized trajectory parameters")
    plt.title("Global trajectory policy fit")
    plt.grid(True, linestyle="--", alpha=0.6)
    _maybe_save(fig, save_dir, "policy_fit_loss.png", show)


def plot_evaluation_summary(rows: Sequence[Dict[str, object]], save_dir: Optional[str] = None, show: bool = True):
    if not rows:
        return
    labels = [f"{r['case'].theta_goal_deg:.0f}\n{r['case'].alpha_deg:.0f}/{r['case'].phi_deg:.0f}" for r in rows]
    x = np.arange(len(rows))
    existing = np.array([r["existing_metrics"]["total_cost"] for r in rows], dtype=float)
    learned = np.array([r["rl_metrics"]["total_cost"] for r in rows], dtype=float)
    fig = plt.figure(figsize=(max(10, len(rows) * 0.55), 5))
    width = 0.4
    plt.bar(x - width/2, existing, width, label="fixed-horizon LQR rollout")
    plt.bar(x + width/2, learned, width, label="NN/GPS reference")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("closed-loop rollout objective cost")
    plt.title("Cost comparison")
    plt.legend(); plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    _maybe_save(fig, save_dir, "cost_comparison.png", show)

#%%
def run_full_trajectory_rl_experiment(
    train_goal_degs: Sequence[float] = (70, 80),
    test_goal_degs: Sequence[float] = (60, 75),
    tilt_degs: Sequence[float] = (0,5, 8, 15, 20),
    cem_iters: int = 8,
    population: int = 24,
    policy_epochs: int = 1000,
    save_dir: str = "trajectory_rl_results",
    show_plots: bool = True,
    resume_training: bool = False,
    skip_existing_cases: bool = False,
):
    traj_cfg, obj_cfg = make_default_configs()
    train_cases = make_cases(train_goal_degs, tilt_degs, coupled_tilt=True)
    train_output = train_gps_trajectory_policy(train_cases, traj_cfg, obj_cfg, cem_iters=cem_iters, population=population, elite_frac=0.20, policy_epochs=policy_epochs, seed=1, save_dir=save_dir, resume_training=resume_training, skip_existing_cases=skip_existing_cases)
    test_cases = make_cases(test_goal_degs, tilt_degs, coupled_tilt=True)
    rows = evaluate_policy_vs_existing(train_output["policy"], test_cases, traj_cfg, obj_cfg, seed=1000)
    plot_training_progress(train_output, save_dir=save_dir, show=show_plots)
    plot_evaluation_summary(rows, save_dir=save_dir, show=show_plots)
    with open(os.path.join(save_dir, "evaluation_summary.json"), "w") as f:
        json.dump([dict(theta_goal_deg=r["case"].theta_goal_deg, alpha_deg=r["case"].alpha_deg, phi_deg=r["case"].phi_deg, existing_cost=r["existing_metrics"]["total_cost"], rl_cost=r["rl_metrics"]["total_cost"], improvement_ratio=r["improvement_ratio"]) for r in rows], f, indent=2)
    return dict(train_output=train_output, evaluation_rows=rows, traj_cfg=traj_cfg, obj_cfg=obj_cfg)


#%% ========================= RUN IN SPYDER =========================
results = run_full_trajectory_rl_experiment(
    cem_iters=10,
    population=50,
    policy_epochs=3000,
    save_dir="trajectory_rl_results_smoke",
    show_plots=True,
    resume_training=True,
)
# To continue training from saved weights, set resume_training=True.

#%% ========================= INLINE SINGLE GOAL COMPARISON =========================

# Choose your single test condition here
theta_goal_deg = 60.0
alpha_deg = 10.0
phi_deg = 0.0

save_dir = "trajectory_rl_results_smoke"
model_filename = "trajectory_policy.pt"
show_plots = True

# Build configs
traj_cfg, obj_cfg = make_default_configs()

# Build one case
case = CaseSpec(
    theta_goal=math.radians(theta_goal_deg),
    alpha=math.radians(alpha_deg),
    phi=math.radians(phi_deg),
)

print("\n================ SINGLE GOAL COMPARISON ================")
print(f"theta_goal = {theta_goal_deg:.2f} deg")
print(f"alpha      = {alpha_deg:.2f} deg")
print(f"phi        = {phi_deg:.2f} deg")

# ---------------------------------------------------------
# 1) Load trained NN/GPS trajectory policy
# ---------------------------------------------------------
model_path = os.path.join(save_dir, model_filename)

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Could not find trained trajectory policy at:\n{model_path}\n\n"
        "First run training, or set save_dir/model_filename correctly."
    )

policy = TrajectoryPolicyNet(traj_cfg)

with np.load(model_path, allow_pickle=False) as data:
    policy.load_state_dict({k: data[k] for k in data.files})

print(f"Loaded NN/GPS trajectory policy from: {model_path}")

# ---------------------------------------------------------
# 2) Generate NN/GPS reference trajectory
# ---------------------------------------------------------
plant_p, _, _, _, _ = system_for_case(case)

gps_params = policy_params(policy, case)

_, gps_theta_ref, gps_omega_ref, gps_T = build_monotone_reference(
    theta_goal=case.theta_goal,
    params=gps_params,
    dt=plant_p.dt,
    cfg=traj_cfg,
    theta0=0.0,
)

print(f"Generated NN/GPS reference duration: {gps_T:.4f} s")

# ---------------------------------------------------------
# 3) Track NN/GPS reference using LQR/TDE/SMC system
# ---------------------------------------------------------
gps_raw_metrics, gps_logs, gps_plant_p = simulate_reference(
    case=case,
    theta_ref=gps_theta_ref,
    duration=gps_T,
    seed=123,
    theta0=0.0,
)

gps_metrics = trajectory_objective(
    logs=gps_logs,
    theta_goal=case.theta_goal,
    plant_p=gps_plant_p,
    obj_cfg=obj_cfg,
)

# ---------------------------------------------------------
# 4) Track original LQR reference
# ---------------------------------------------------------
lqr_raw_metrics, lqr_logs, lqr_theta_ref, lqr_T, lqr_plant_p = simulate_existing_reference(
    case=case,
    seed=123,
    theta0=0.0,
)

lqr_metrics = trajectory_objective(
    logs=lqr_logs,
    theta_goal=case.theta_goal,
    plant_p=lqr_plant_p,
    obj_cfg=obj_cfg,
)

# ---------------------------------------------------------
# 5) Print comparison summary
# ---------------------------------------------------------
lqr_cost = float(lqr_metrics["total_cost"])
gps_cost = float(gps_metrics["total_cost"])

improvement = lqr_cost - gps_cost
improvement_ratio = improvement / max(abs(lqr_cost), 1e-12)

print("\n---------------- RESULTS ----------------")
print(f"Fixed-horizon LQR closed-loop rollout objective cost = {lqr_cost:.6g}")
print(f"NN/GPS closed-loop rollout objective cost            = {gps_cost:.6g}")
print(f"Improvement        = {improvement:.6g}")
print(f"Improvement ratio  = {100.0 * improvement_ratio:.2f} %")

print("\nFinal theta error:")
print(f"LQR    = {lqr_metrics['final_theta_error']:.6g} rad "
      f"({math.degrees(lqr_metrics['final_theta_error']):.4f} deg)")
print(f"NN/GPS = {gps_metrics['final_theta_error']:.6g} rad "
      f"({math.degrees(gps_metrics['final_theta_error']):.4f} deg)")

print("\nMax values:")
print(f"LQR max |u_command|     = {lqr_metrics['max_abs_u_total']:.6g}")
print(f"GPS max |u_command|     = {gps_metrics['max_abs_u_total']:.6g}")
print(f"LQR max |tau_m|         = {lqr_metrics['max_abs_tau_m']:.6g}")
print(f"GPS max |tau_m|         = {gps_metrics['max_abs_tau_m']:.6g}")

# ---------------------------------------------------------
# 6) Extract logs safely
# ---------------------------------------------------------
t_lqr = np.asarray(lqr_logs["t"], dtype=float)
theta_lqr = np.asarray(lqr_logs["theta"], dtype=float)
theta_ref_lqr = np.asarray(lqr_logs["theta_ref"], dtype=float)
omega_lqr = np.asarray(lqr_logs["omega"], dtype=float)
omega_ref_lqr = np.asarray(lqr_logs["omega_ref"], dtype=float)
u_lqr = np.asarray(lqr_logs["u_total"], dtype=float)
tau_lqr = np.asarray(lqr_logs.get("tau_m", lqr_logs["u_total"]), dtype=float)

t_gps = np.asarray(gps_logs["t"], dtype=float)
theta_gps = np.asarray(gps_logs["theta"], dtype=float)
theta_ref_gps = np.asarray(gps_logs["theta_ref"], dtype=float)
omega_gps = np.asarray(gps_logs["omega"], dtype=float)
omega_ref_gps = np.asarray(gps_logs["omega_ref"], dtype=float)
u_gps = np.asarray(gps_logs["u_total"], dtype=float)
tau_gps = np.asarray(gps_logs.get("tau_m", gps_logs["u_total"]), dtype=float)

if "tau_m" not in lqr_logs:
    print("Warning: tau_m not found in LQR logs. Used u_total as fallback.")

if "tau_m" not in gps_logs:
    print("Warning: tau_m not found in NN/GPS logs. Used u_total as fallback.")

# ---------------------------------------------------------
# 7) Plot theta
# ---------------------------------------------------------
fig = plt.figure(figsize=(9, 5))
plt.plot(t_lqr, theta_lqr, label="LQR actual theta")
plt.plot(t_lqr, theta_ref_lqr, "--", label="LQR reference theta")
plt.plot(t_gps, theta_gps, label="NN/GPS actual theta")
plt.plot(t_gps, theta_ref_gps, "--", label="NN/GPS reference theta")
plt.xlabel("Time [s]")
plt.ylabel("Theta [rad]")
plt.title("Theta tracking comparison")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "single_goal_theta.png"), dpi=180)
if show_plots:
    plt.show()
else:
    plt.close(fig)

# ---------------------------------------------------------
# 8) Plot theta dot / omega
# ---------------------------------------------------------
fig = plt.figure(figsize=(9, 5))
plt.plot(t_lqr, omega_lqr, label="LQR actual omega")
plt.plot(t_lqr, omega_ref_lqr, "--", label="LQR reference omega")
plt.plot(t_gps, omega_gps, label="NN/GPS actual omega")
plt.plot(t_gps, omega_ref_gps, "--", label="NN/GPS reference omega")
plt.xlabel("Time [s]")
plt.ylabel("Omega / theta_dot [rad/s]")
plt.title("Angular velocity tracking comparison")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "single_goal_omega.png"), dpi=180)
if show_plots:
    plt.show()
else:
    plt.close(fig)

# ---------------------------------------------------------
# 9) Plot u command
# ---------------------------------------------------------
fig = plt.figure(figsize=(9, 5))
plt.plot(t_lqr, u_lqr, label="LQR u_command")
plt.plot(t_gps, u_gps, label="NN/GPS u_command")
plt.xlabel("Time [s]")
plt.ylabel("u_command")
plt.title("Command input comparison")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "single_goal_u_command.png"), dpi=180)
if show_plots:
    plt.show()
else:
    plt.close(fig)

# ---------------------------------------------------------
# 10) Plot induced torque tau_m
# ---------------------------------------------------------
fig = plt.figure(figsize=(9, 5))
plt.plot(t_lqr, tau_lqr, label="LQR induced torque tau_m")
plt.plot(t_gps, tau_gps, label="NN/GPS induced torque tau_m")
plt.xlabel("Time [s]")
plt.ylabel("tau_m")
plt.title("Induced torque comparison")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "single_goal_tau_m.png"), dpi=180)
if show_plots:
    plt.show()
else:
    plt.close(fig)

print("\nSaved figures:")
print(os.path.join(save_dir, "single_goal_theta.png"))
print(os.path.join(save_dir, "single_goal_omega.png"))
print(os.path.join(save_dir, "single_goal_u_command.png"))
print(os.path.join(save_dir, "single_goal_tau_m.png"))

print("========================================================\n")


