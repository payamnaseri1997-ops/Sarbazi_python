# -*- coding: utf-8 -*-
"""
true_gps_lqr_guided_addon.py

True GPS-style trajectory-learning add-on for LQR_TrjOPt_TDESMCwithRLresidual.py.

IMPORTANT DESIGN CHOICE
-----------------------
This file does NOT change your plant dynamics or your TDE+SMC controller.
It only changes the reference trajectory supplied to the controller.

The learned reference is explicitly guided by the existing LQR/iLQR reference:

    theta_ref_gps(t) = LQR-guide-shape(tau) + learned residual deformation

implemented in velocity-profile form:

    v_gps(z) = v_lqr_guide(z) * exp(residual_basis(z) @ params)
    theta_ref_gps(z) = integral(v_gps) normalized to exactly reach theta_goal

So this is not "trajectory from scratch". The LQR trajectory is the guide/teacher,
and the deep policy learns residual parameters around it.

Workflow
--------
1) For many training cases (theta_goal, alpha tilt, phi tilt), create the existing
   LQR reference.
2) Run a local CEM trajectory optimizer initialized around the LQR guide.
3) Train a neural network policy to map:
       [theta_goal, alpha, phi] -> residual trajectory parameters
4) For new goals/tilts, the policy predicts a guided residual trajectory.
5) Optionally do a small GPS refinement around the policy output.
6) Compare existing LQR reference vs NN reference vs GPS-refined reference.

Objective
---------
Minimize:
    controller command energy
  + theta tracking error
  + velocity tracking error
  + final theta/velocity error
  + constraint penalties:
        theta_final = theta_goal
        |theta_dot| <= 0.2 rad/s
        |tau_m| <= 2000 N.m
        |u_total| <= 500

Usage
-----
Place this file next to LQR_TrjOPt_TDESMCwithRLresidual.py and run:

from true_gps_lqr_guided_addon import run_true_gps_experiment
results = run_true_gps_experiment()

For a quick smoke test:
results = run_true_gps_experiment(
    train_goal_degs=(30, 90, 150),
    test_goal_degs=(45, 135),
    tilt_degs=(0, 20),
    cem_iters=3,
    population=12,
    policy_epochs=300,
    refinement_iters=2,
)
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple, Optional, Sequence
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


# The original advanced module imports rl_simple/rl_sac at top level for its own
# residual-control training utilities. This true-GPS add-on never uses those
# agents, but the import can fail in folders where rl_simple.py is absent.
# Provide a tiny harmless stub only when the real module is unavailable.
import sys
import types
import importlib.util
if importlib.util.find_spec("rl_simple") is None and "rl_simple" not in sys.modules:
    _stub = types.ModuleType("rl_simple")
    class _StubRLConfig:
        def __init__(self, *args, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    class _StubSimpleResidualPolicy:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("rl_simple.py is not available; SimpleResidualPolicy cannot be used.")
    _stub.RLConfig = _StubRLConfig
    _stub.SimpleResidualPolicy = _StubSimpleResidualPolicy
    sys.modules["rl_simple"] = _stub

# Your existing system. This add-on calls these; it does not edit them.
# The advanced file in different project folders has sometimes been named either
# LQR_TrjOPt_TDESMCwithRLresidual.py or LQR_TrjOPt_TDESMCwithRLresidual_2.py.
# We require the advanced version because it contains actuator dynamics, tau_i,
# K_t, geometry/mass parameters, custom-reference rollout_once(), and tau_m logs.
import importlib

_SYSTEM_MODULE = None
_SYSTEM_IMPORT_ERRORS = []
for _module_name in (
    "LQR_TrjOPt_TDESMCwithRLresidual_2",
    "LQR_TrjOPt_TDESMCwithRLresidual",
):
    try:
        _m = importlib.import_module(_module_name)
        _ann = getattr(getattr(_m, "PlantParams", None), "__annotations__", {})
        if hasattr(_m, "rollout_once") and "tau_i" in _ann and "K_t" in _ann:
            _SYSTEM_MODULE = _m
            break
        _SYSTEM_IMPORT_ERRORS.append(f"{_module_name}: imported but not the advanced actuator/geometry version")
    except Exception as _exc:
        _SYSTEM_IMPORT_ERRORS.append(f"{_module_name}: {_exc}")

if _SYSTEM_MODULE is None:
    raise ImportError(
        "Could not import the advanced LQR/TDE+SMC module. Expected a module named "
        "LQR_TrjOPt_TDESMCwithRLresidual_2 or LQR_TrjOPt_TDESMCwithRLresidual "
        "with PlantParams(tau_i, K_t, geometry fields) and rollout_once().\n"
        + "\n".join(_SYSTEM_IMPORT_ERRORS)
    )

PlantParams = _SYSTEM_MODULE.PlantParams
NominalModel = _SYSTEM_MODULE.NominalModel
LQRWeights = _SYSTEM_MODULE.LQRWeights
SMCConfig = _SYSTEM_MODULE.SMCConfig
CostConfig = _SYSTEM_MODULE.CostConfig
Task = _SYSTEM_MODULE.Task
OneDOFRotorPlant = _SYSTEM_MODULE.OneDOFRotorPlant
rollout_once = _SYSTEM_MODULE.rollout_once
finite_diff = _SYSTEM_MODULE.finite_diff
build_AB = _SYSTEM_MODULE.build_AB
finite_horizon_lqr_gain = _SYSTEM_MODULE.finite_horizon_lqr_gain


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def trapz_safe(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral that does not rely on np.trapz/np.trapezoid."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(y) < 2:
        return 0.0
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def cumtrapz_safe(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral with y[0] integral = 0."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(y, dtype=float)
    if len(y) >= 2:
        out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * np.diff(x))
    return out


def sat(x: float, limit: float) -> float:
    return max(-limit, min(limit, float(x)))


def mkdir(path: Optional[str]):
    if path:
        os.makedirs(path, exist_ok=True)


# -----------------------------------------------------------------------------
# Configurations
# -----------------------------------------------------------------------------

@dataclass
class GPSObjectiveConfig:
    """Trajectory objective and hard-constraint penalties."""
    omega_limit: float = 0.2          # rad/s, actual and reference velocity limit
    torque_limit: float = 2000.0      # N.m, physical shaft torque tau_m
    command_limit: float = 500.0      # command saturation u_total

    # Objective weights. Constraint penalties intentionally dominate.
    w_energy: float = 1.0             # integral (u_total / command_limit)^2 dt
    w_theta_track: float = 5.0        # integral normalized theta tracking error^2 dt
    w_omega_track: float = 1.0        # integral normalized velocity tracking error^2 dt
    w_final_theta: float = 5.0e4      # final theta equality constraint
    w_final_omega: float = 5.0e3      # final stop condition
    w_ref_omega_violation: float = 1.0e6
    w_actual_omega_violation: float = 1.0e6
    w_tau_violation: float = 1.0e5
    w_command_violation: float = 1.0e5
    w_guide_theta: float = 0.05       # stay near LQR guide shape, but not too strongly
    w_guide_omega: float = 0.01
    w_duration: float = 0.001         # mild preference for shorter feasible trajectories


@dataclass
class GPSTrajectoryConfig:
    """Guided residual trajectory parameterization."""
    n_basis: int = 6
    omega_ref_limit: float = 0.2
    duration_margin: float = 1.05     # T >= margin * minimum duration implied by omega limit
    max_duration_factor: float = 2.5  # T <= max_duration_factor * T_min, unless min_extra_duration dominates
    min_extra_duration: float = 5.0
    duration_exp_scale: float = 0.35
    z_grid_size: int = 1201
    hidden_sizes: Tuple[int, int] = (96, 96)


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


# -----------------------------------------------------------------------------
# Your requested plant/model parameters
# -----------------------------------------------------------------------------

def requested_project_params(alpha: float = 0.0, phi: float = 0.0):
    """Return requested plant/model/controller parameters.

    The TDE+SMC controller law is not changed. Only its input reference changes.
    """
    plant_p = PlantParams(
        J=14099.0,
        b=0.09,
        u_max=500.0,
        omega_max=8.0,
        dt=0.002,
        tau_i=0.1,
        K_t=5.0,
        m1=5000.0,
        R1=3.10,
        m2=600.5,
        a2=1.5,
        r1=0.5,
        m3=600.8,
        a3=1.5,
        r2=0.5,
        m4=300.4,
        L4=1.20,
        l4_c=0.60,
        gamma=0.0,
        beta2=0.0,
        beta4=0.0,
        alpha=alpha,
        phi=phi,
        g=9.81,
        subtract_gravity_in_ueq=False,
    )

    # Nominal model used by your existing LQR and TDE+SMC computations.
    nom = NominalModel(J=14099.0, b=0.09)

    lqr_w = LQRWeights(
        q_theta=85.0,
        q_omega=18.0,
        r_u=0.02,
        qT_theta=4200.0,
        qT_omega=220.0,
        omega_limit_penalty=1000.0,
    )

    # Keep your controller structure. These are gains, not a controller-law change.
    smc_cfg = SMCConfig(lambda_s=35.0, k=0.85, phi=0.025)

    cost_cfg = CostConfig(
        w_e=8.0,
        w_edot=1.0,
        w_u=0.03,
        w_omega=0.3,
        goal_tol=1e-2,
        done_bonus=2.0,
    )
    return plant_p, nom, lqr_w, smc_cfg, cost_cfg


# -----------------------------------------------------------------------------
# Existing LQR guide reference
# -----------------------------------------------------------------------------

def existing_lqr_horizon(theta0: float, theta_goal: float) -> float:
    """Match your original idea: pi rad move takes 4 seconds."""
    delta = abs(theta_goal - theta0)
    if delta < 1e-12:
        return 0.5
    return 4.0 * delta / math.pi


def generate_existing_lqr_reference(
    nom: NominalModel,
    plant_p: PlantParams,
    lqr_w: LQRWeights,
    theta0: float,
    theta_goal: float,
    dt: float,
) -> Tuple[np.ndarray, float]:
    """Local scalar-safe copy of your LQR/iLQR-like reference generator."""
    T = existing_lqr_horizon(theta0, theta_goal)
    N = max(1, int(round(T / dt)))
    T = N * dt
    A, B = build_AB(nom, dt)
    Ks = finite_horizon_lqr_gain(A, B, N, lqr_w)

    x = np.array([theta0, 0.0], dtype=float)
    x_goal = np.array([theta_goal, 0.0], dtype=float)
    x_ref = np.zeros((N + 1, 2), dtype=float)
    x_ref[0] = x

    for k in range(N):
        x_tilde = x - x_goal
        excess = abs(x[1]) - getattr(plant_p, "omega_max", 1e9)
        if excess > 0 and hasattr(lqr_w, "omega_limit_penalty"):
            x_tilde[1] += lqr_w.omega_limit_penalty * excess * math.copysign(1.0, x[1])
        u = float((-Ks[k] @ x_tilde.reshape(2, 1)).item())
        u = sat(u, plant_p.u_max)
        x = A @ x + B.flatten() * u
        x_ref[k + 1] = x

    theta_ref = x_ref[:, 0].copy()
    theta_ref[0] = theta0
    # Do not force theta_ref[-1] = theta_goal here. For heavy plants, the
    # finite-horizon linear reference may not reach the goal; forcing the last
    # sample creates an artificial jump and a huge velocity spike. The guide
    # shape is normalized safely in lqr_guide_shape().
    return theta_ref, T


def lqr_guide_shape(theta_ref: np.ndarray, theta_goal: float, n_grid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert existing LQR reference to normalized guide position h(z) and velocity shape v(z).

    h(z) is monotone from 0 to 1. v(z) integrates to 1.
    """
    z_old = np.linspace(0.0, 1.0, len(theta_ref))
    z_grid = np.linspace(0.0, 1.0, n_grid)
    if abs(theta_goal) < 1e-12:
        h = np.zeros_like(z_grid)
        v = np.ones_like(z_grid)
        v /= max(trapz_safe(v, z_grid), 1e-12)
        return z_grid, h, v

    raw = np.asarray(theta_ref, dtype=float) - float(theta_ref[0])
    # Normalize by the LQR reference's own final progress, not by theta_goal.
    # This preserves the LQR shape even if the heavy-system LQR horizon did not
    # reach the goal, and avoids a discontinuous endpoint jump.
    scale = float(raw[-1])
    if abs(scale) < 1e-9:
        # Degenerate guide: fall back to a standard minimum-jerk shape.
        h = 10.0 * z_grid**3 - 15.0 * z_grid**4 + 6.0 * z_grid**5
    else:
        h_old = raw / scale
        h_old = np.clip(h_old, 0.0, 1.0)
        h_old = np.maximum.accumulate(h_old)
        h_old[0] = 0.0
        h_old[-1] = 1.0
        h = np.interp(z_grid, z_old, h_old)
        h = np.maximum.accumulate(np.clip(h, 0.0, 1.0))
        h[0] = 0.0
        h[-1] = 1.0
        # Lightly blend with minimum-jerk to remove numerical spikes while still
        # keeping the LQR guide as the dominant teacher.
        h_mj = 10.0 * z_grid**3 - 15.0 * z_grid**4 + 6.0 * z_grid**5
        h = 0.85 * h + 0.15 * h_mj
        h = np.maximum.accumulate(np.clip(h, 0.0, 1.0))
        h[0] = 0.0
        h[-1] = 1.0

    v = np.gradient(h, z_grid)
    v = np.maximum(v, 1e-6)
    # Clip extreme derivative spikes caused by aggressive finite-horizon LQR
    # references. This keeps the guide useful under the |theta_dot| constraint.
    cap = max(5.0, 3.0 * float(np.percentile(v, 95)))
    v = np.minimum(v, cap)
    area = max(trapz_safe(v, z_grid), 1e-12)
    v /= area
    return z_grid, h, v


# -----------------------------------------------------------------------------
# Guided residual trajectory parameterization
# -----------------------------------------------------------------------------

def residual_basis(z: np.ndarray, n_basis: int) -> np.ndarray:
    """Endpoint-safe basis for residual deformation of guide velocity shape."""
    cols = []
    for k in range(1, n_basis + 1):
        # Sin terms vanish at z=0,1 and are smooth.
        cols.append(np.sin(k * math.pi * z))
    return np.stack(cols, axis=1)


def decode_duration(raw_duration: float, theta_goal: float, T_guide: float, cfg: GPSTrajectoryConfig) -> float:
    delta = abs(theta_goal)
    if delta < 1e-12:
        return max(0.5, T_guide)
    T_min = cfg.duration_margin * delta / cfg.omega_ref_limit
    T_center = max(T_guide, T_min)
    T_max = max(cfg.max_duration_factor * T_min, T_min + cfg.min_extra_duration, T_center)
    T = T_center * math.exp(cfg.duration_exp_scale * float(raw_duration))
    return float(np.clip(T, T_min, T_max))


def build_guided_residual_reference(
    theta_goal: float,
    params: np.ndarray,
    base_theta_ref: np.ndarray,
    T_base: float,
    dt: float,
    cfg: GPSTrajectoryConfig,
    theta0: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Dict[str, np.ndarray]]:
    """Build theta_ref = LQR-guide-shape + learned residual deformation.

    Params length = n_basis + 1. The first n_basis values deform the LQR guide
    velocity shape. The last value changes duration.
    """
    params = np.asarray(params, dtype=float).reshape(-1)
    expected = cfg.n_basis + 1
    if params.size != expected:
        raise ValueError(f"params length must be {expected}, got {params.size}")

    delta = float(theta_goal - theta0)
    if abs(delta) < 1e-12:
        T = max(0.5, T_base)
        N = max(1, int(math.ceil(T / dt)))
        T = N * dt
        t = np.arange(N + 1, dtype=float) * dt
        theta_ref = np.full(N + 1, theta0, dtype=float)
        omega_ref = np.zeros(N + 1, dtype=float)
        extra = dict(z=np.linspace(0.0, 1.0, cfg.z_grid_size), h_base=np.zeros(cfg.z_grid_size), h_gps=np.zeros(cfg.z_grid_size), v_base=np.ones(cfg.z_grid_size), v_gps=np.ones(cfg.z_grid_size))
        return t, theta_ref, omega_ref, T, extra

    z, h_base, v_base = lqr_guide_shape(base_theta_ref, theta_goal, cfg.z_grid_size)
    B = residual_basis(z, cfg.n_basis)
    residual_log = B @ params[:cfg.n_basis]
    residual_log = np.clip(residual_log, -3.0, 3.0)

    # True guided residual: deform the LQR guide velocity shape, do not replace it.
    v_gps = np.maximum(v_base, 1e-8) * np.exp(residual_log)
    v_gps = np.maximum(v_gps, 1e-9)
    area = max(trapz_safe(v_gps, z), 1e-12)
    v_gps = v_gps / area

    # Duration: start from the safe version of the base LQR guide.
    T = decode_duration(params[-1], theta_goal=theta_goal, T_guide=T_base, cfg=cfg)

    # Enforce reference velocity limit by increasing duration if the shape has a high peak.
    shape_peak = float(np.max(v_gps))
    T_needed = cfg.duration_margin * abs(delta) * shape_peak / cfg.omega_ref_limit
    T = max(T, T_needed)
    N = max(1, int(math.ceil(T / dt)))
    T = N * dt

    h_gps = cumtrapz_safe(v_gps, z)
    if h_gps[-1] <= 0:
        h_gps = h_base.copy()
    else:
        h_gps /= h_gps[-1]
    h_gps = np.clip(np.maximum.accumulate(h_gps), 0.0, 1.0)
    h_gps[0] = 0.0
    h_gps[-1] = 1.0

    t = np.arange(N + 1, dtype=float) * dt
    z_time = t / max(T, 1e-12)
    h_time = np.interp(z_time, z, h_gps)
    theta_ref = theta0 + delta * h_time
    theta_ref[0] = theta0
    theta_ref[-1] = theta_goal
    omega_ref = finite_diff(theta_ref, dt)

    extra = dict(z=z, h_base=h_base, h_gps=h_gps, v_base=v_base, v_gps=v_gps)
    return t, theta_ref, omega_ref, T, extra


# -----------------------------------------------------------------------------
# Rollout and objective
# -----------------------------------------------------------------------------

def simulate_reference(
    case: GPSCase,
    theta_ref: np.ndarray,
    T: float,
    seed: int = 0,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Use your existing rollout_once with a custom reference. Controller unchanged."""
    plant_p, nom, lqr_w, smc_cfg, cost_cfg = requested_project_params(case.alpha, case.phi)
    task = Task(theta0=0.0, omega0=0.0, theta_goal=case.theta_goal)
    plant = OneDOFRotorPlant(plant_p)

    metrics, logs = rollout_once(
        plant=plant,
        nom=nom,
        task=task,
        lqr_w=lqr_w,
        smc_cfg=smc_cfg,
        agent=None,
        cost_cfg=cost_cfg,
        reference={"kind": "manual", "theta": np.asarray(theta_ref, dtype=float), "duration": float(T)},
        seed=seed,
        collect_logs=True,
    )
    return metrics, logs


def compute_objective(
    logs: Dict[str, np.ndarray],
    case: GPSCase,
    obj: GPSObjectiveConfig,
    guide_theta_ref: Optional[np.ndarray] = None,
    guide_T: Optional[float] = None,
) -> Dict[str, float]:
    """Compute trajectory objective and constraint metrics."""
    t = np.asarray(logs["t"], dtype=float)
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 0.002
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

    theta_err = theta - theta_ref
    omega_err = omega - omega_ref

    energy = float(np.sum((u_total / cmd_scale) ** 2) * dt)
    theta_track = float(np.sum((theta_err / goal_scale) ** 2) * dt)
    omega_track = float(np.sum((omega_err / omega_scale) ** 2) * dt)

    final_theta_error = float(theta[-1] - case.theta_goal)
    final_omega = float(omega[-1])
    final_cost = (final_theta_error / goal_scale) ** 2 + (final_omega / omega_scale) ** 2

    ref_omega_violation = np.maximum(0.0, np.abs(omega_ref) - obj.omega_limit)
    actual_omega_violation = np.maximum(0.0, np.abs(omega) - obj.omega_limit)
    tau_violation = np.maximum(0.0, np.abs(tau_m) - obj.torque_limit)
    command_violation = np.maximum(0.0, np.abs(u_total) - obj.command_limit)

    ref_omega_violation_cost = float(np.sum((ref_omega_violation / omega_scale) ** 2) * dt)
    actual_omega_violation_cost = float(np.sum((actual_omega_violation / omega_scale) ** 2) * dt)
    tau_violation_cost = float(np.sum((tau_violation / torque_scale) ** 2) * dt)
    command_violation_cost = float(np.sum((command_violation / cmd_scale) ** 2) * dt)

    guide_theta_cost = 0.0
    guide_omega_cost = 0.0
    if guide_theta_ref is not None and guide_T is not None:
        # Compare guide and candidate on normalized time, so the guide can be time-stretched.
        z_cand = np.linspace(0.0, 1.0, len(theta_ref))
        z_guide = np.linspace(0.0, 1.0, len(guide_theta_ref))
        guide_resampled = np.interp(z_cand, z_guide, np.asarray(guide_theta_ref, dtype=float))
        guide_omega = finite_diff(guide_resampled, dt)
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
        total=float(total),
        energy=energy,
        theta_track=theta_track,
        omega_track=omega_track,
        final_theta_error=abs(final_theta_error),
        final_omega_abs=abs(final_omega),
        max_abs_omega=float(np.max(np.abs(omega))) if len(omega) else 0.0,
        max_abs_omega_ref=float(np.max(np.abs(omega_ref))) if len(omega_ref) else 0.0,
        max_abs_tau_m=float(np.max(np.abs(tau_m))) if len(tau_m) else 0.0,
        max_abs_u_total=float(np.max(np.abs(u_total))) if len(u_total) else 0.0,
        ref_omega_violation=ref_omega_violation_cost,
        actual_omega_violation=actual_omega_violation_cost,
        tau_violation=tau_violation_cost,
        command_violation=command_violation_cost,
        guide_theta=guide_theta_cost,
        guide_omega=guide_omega_cost,
        duration=duration,
    )


def evaluate_existing_case(case: GPSCase, obj: GPSObjectiveConfig, seed: int = 0):
    plant_p, nom, lqr_w, _, _ = requested_project_params(case.alpha, case.phi)
    theta_ref, T = generate_existing_lqr_reference(nom, plant_p, lqr_w, 0.0, case.theta_goal, plant_p.dt)
    _, logs = simulate_reference(case, theta_ref, T, seed=seed)
    metrics = compute_objective(logs, case, obj)
    return metrics, logs, theta_ref, T


def evaluate_guided_params(
    case: GPSCase,
    params: np.ndarray,
    traj_cfg: GPSTrajectoryConfig,
    obj_cfg: GPSObjectiveConfig,
    seed: int = 0,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], np.ndarray, float, Dict[str, np.ndarray], np.ndarray, float]:
    plant_p, nom, lqr_w, _, _ = requested_project_params(case.alpha, case.phi)
    base_ref, base_T = generate_existing_lqr_reference(nom, plant_p, lqr_w, 0.0, case.theta_goal, plant_p.dt)
    _, theta_ref, _, T, extra = build_guided_residual_reference(
        theta_goal=case.theta_goal,
        params=params,
        base_theta_ref=base_ref,
        T_base=base_T,
        dt=plant_p.dt,
        cfg=traj_cfg,
        theta0=0.0,
    )
    _, logs = simulate_reference(case, theta_ref, T, seed=seed)
    metrics = compute_objective(logs, case, obj_cfg, guide_theta_ref=base_ref, guide_T=base_T)
    return metrics, logs, theta_ref, T, extra, base_ref, base_T


# -----------------------------------------------------------------------------
# Local trajectory optimizer (GPS local teacher)
# -----------------------------------------------------------------------------

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
    """CEM optimizer initialized around LQR residual = zero.

    The zero parameter vector means: use the LQR guide shape, only time-scaled
    enough to satisfy the velocity limit.
    """
    rng = np.random.default_rng(seed)
    dim = traj_cfg.n_basis + 1
    mean = np.zeros(dim, dtype=float) if init_mean is None else np.asarray(init_mean, dtype=float).copy()
    std = np.ones(dim, dtype=float) * 0.65 if init_std is None else np.asarray(init_std, dtype=float).copy()
    # Duration tends to need less exploration than shape coefficients.
    std[-1] = min(std[-1], 0.45)

    elite_n = max(2, int(round(population * elite_frac)))
    best = dict(cost=np.inf, params=None, metrics=None, logs=None, theta_ref=None, T=None, extra=None)
    history: List[Dict[str, float]] = []

    for it in range(cem_iters):
        candidates = rng.normal(mean, std, size=(population, dim))
        # Always include current mean and zero-guide candidate for safety.
        candidates[0] = mean
        if population > 1:
            candidates[1] = np.zeros(dim)

        scored = []
        for j, p in enumerate(candidates):
            try:
                metrics, logs, theta_ref, T, extra, base_ref, base_T = evaluate_guided_params(
                    case, p, traj_cfg, obj_cfg, seed=seed + 1000 * it + j
                )
                cost = metrics["total"]
            except Exception as exc:
                metrics, logs, theta_ref, T, extra = None, None, None, None, None
                cost = np.inf
            scored.append((cost, p, metrics, logs, theta_ref, T, extra))

        scored.sort(key=lambda x: x[0])
        elites = np.stack([s[1] for s in scored[:elite_n]], axis=0)
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + 1e-4
        std = np.minimum(std, 1.25)
        std[-1] = min(std[-1], 0.60)

        if scored[0][0] < best["cost"]:
            cost, p, metrics, logs, theta_ref, T, extra = scored[0]
            best.update(cost=float(cost), params=p.copy(), metrics=metrics, logs=logs, theta_ref=theta_ref, T=T, extra=extra)

        history.append(dict(iteration=it, best=float(scored[0][0]), mean=float(np.mean([s[0] for s in scored if np.isfinite(s[0])]))))
        print(f"CEM {case.label()} iter {it+1:02d}/{cem_iters}: best={scored[0][0]:.6g}")

    return dict(case=case, best=best, final_mean=mean, final_std=std, history=history)


# -----------------------------------------------------------------------------
# Deep GPS policy: maps goal/tilt to LQR-residual trajectory parameters
# -----------------------------------------------------------------------------

class GuidedTrajectoryPolicy(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Tuple[int, int]):
        super().__init__()
        layers: List[nn.Module] = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def case_features(case: GPSCase) -> np.ndarray:
    """Normalize inputs for goal-conditioned policy."""
    return np.array([
        case.theta_goal / math.pi,     # 0..1 for 0..180deg
        case.alpha / math.radians(20), # 0..1 for 0..20deg
        case.phi / math.radians(20),
    ], dtype=np.float32)


def train_policy_from_local_teachers(
    local_results: Sequence[Dict[str, object]],
    traj_cfg: GPSTrajectoryConfig,
    epochs: int = 1500,
    lr: float = 1e-3,
    seed: int = 0,
) -> Tuple[GuidedTrajectoryPolicy, List[float]]:
    if torch is None:
        raise RuntimeError(f"PyTorch import failed: {_TORCH_IMPORT_ERROR}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = []
    Y = []
    for res in local_results:
        case: GPSCase = res["case"]
        params = np.asarray(res["best"]["params"], dtype=np.float32)
        X.append(case_features(case))
        Y.append(params)
    X_t = torch.as_tensor(np.stack(X, axis=0), dtype=torch.float32)
    Y_t = torch.as_tensor(np.stack(Y, axis=0), dtype=torch.float32)

    policy = GuidedTrajectoryPolicy(3, traj_cfg.n_basis + 1, traj_cfg.hidden_sizes)
    opt = optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-5)
    losses: List[float] = []

    for ep in range(epochs):
        pred = policy(X_t)
        loss = torch.mean((pred - Y_t) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"policy imitation epoch {ep+1}/{epochs}: loss={losses[-1]:.6g}")
    return policy, losses


def policy_predict_params(policy: GuidedTrajectoryPolicy, case: GPSCase) -> np.ndarray:
    x = torch.as_tensor(case_features(case), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        y = policy(x).squeeze(0).cpu().numpy().astype(float)
    return y


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------

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


def train_true_gps_policy(
    train_cases: Sequence[GPSCase],
    traj_cfg: GPSTrajectoryConfig,
    obj_cfg: GPSObjectiveConfig,
    cem_iters: int = 10,
    population: int = 48,
    elite_frac: float = 0.20,
    policy_epochs: int = 1500,
    seed: int = 0,
    save_dir: Optional[str] = None,
) -> Dict[str, object]:
    """Run local CEM teachers and train global deep GPS policy."""
    mkdir(save_dir)
    local_results = []
    for i, case in enumerate(train_cases):
        print(f"\n=== Local GPS teacher {i+1}/{len(train_cases)}: {case.label()} ===")
        res = cem_optimize_guided_case(
            case=case,
            traj_cfg=traj_cfg,
            obj_cfg=obj_cfg,
            cem_iters=cem_iters,
            population=population,
            elite_frac=elite_frac,
            seed=seed + 100 * i,
        )
        local_results.append(res)

    policy, losses = train_policy_from_local_teachers(
        local_results, traj_cfg, epochs=policy_epochs, seed=seed
    )

    if save_dir:
        torch.save(policy.state_dict(), os.path.join(save_dir, "true_gps_policy.pt"))
        with open(os.path.join(save_dir, "training_teachers.json"), "w") as f:
            json.dump([
                dict(
                    theta_goal_deg=r["case"].theta_goal_deg,
                    alpha_deg=r["case"].alpha_deg,
                    phi_deg=r["case"].phi_deg,
                    best_cost=r["best"]["cost"],
                    best_params=np.asarray(r["best"]["params"]).tolist(),
                )
                for r in local_results
            ], f, indent=2)
    return dict(policy=policy, losses=losses, local_results=local_results)


def evaluate_true_gps_policy(
    policy: GuidedTrajectoryPolicy,
    test_cases: Sequence[GPSCase],
    traj_cfg: GPSTrajectoryConfig,
    obj_cfg: GPSObjectiveConfig,
    refinement_iters: int = 4,
    refinement_population: int = 24,
    seed: int = 1000,
) -> List[Dict[str, object]]:
    """Evaluate existing LQR, NN-only GPS, and GPS-refined trajectory."""
    rows = []
    for i, case in enumerate(test_cases):
        print(f"\n=== Evaluate {i+1}/{len(test_cases)}: {case.label()} ===")
        existing_metrics, existing_logs, existing_ref, existing_T = evaluate_existing_case(case, obj_cfg, seed=seed + i)

        p_nn = policy_predict_params(policy, case)
        nn_metrics, nn_logs, nn_ref, nn_T, nn_extra, base_ref, base_T = evaluate_guided_params(
            case, p_nn, traj_cfg, obj_cfg, seed=seed + i
        )

        # True GPS: use the deep policy as the global guide, then locally refine around it.
        ref_result = cem_optimize_guided_case(
            case=case,
            traj_cfg=traj_cfg,
            obj_cfg=obj_cfg,
            init_mean=p_nn,
            init_std=np.r_[np.ones(traj_cfg.n_basis) * 0.25, 0.20],
            cem_iters=refinement_iters,
            population=refinement_population,
            elite_frac=0.25,
            seed=seed + 5000 + i,
        )
        gps_metrics = ref_result["best"]["metrics"]
        gps_logs = ref_result["best"]["logs"]
        gps_ref = ref_result["best"]["theta_ref"]
        gps_T = ref_result["best"]["T"]
        gps_params = ref_result["best"]["params"]
        gps_extra = ref_result["best"]["extra"]

        print(
            f"existing={existing_metrics['total']:.6g}, "
            f"nn_only={nn_metrics['total']:.6g}, "
            f"gps_refined={gps_metrics['total']:.6g}, "
            f"improvement={100*(existing_metrics['total']-gps_metrics['total'])/max(abs(existing_metrics['total']),1e-12):.2f}%"
        )

        rows.append(dict(
            case=case,
            existing_metrics=existing_metrics,
            existing_logs=existing_logs,
            existing_ref=existing_ref,
            existing_T=existing_T,
            nn_params=p_nn,
            nn_metrics=nn_metrics,
            nn_logs=nn_logs,
            nn_ref=nn_ref,
            nn_T=nn_T,
            nn_extra=nn_extra,
            gps_params=gps_params,
            gps_metrics=gps_metrics,
            gps_logs=gps_logs,
            gps_ref=gps_ref,
            gps_T=gps_T,
            gps_extra=gps_extra,
            base_ref=base_ref,
            base_T=base_T,
            refinement_history=ref_result["history"],
        ))
    return rows


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _save_or_show(fig, save_dir: Optional[str], name: str, show: bool):
    if save_dir:
        mkdir(save_dir)
        fig.savefig(os.path.join(save_dir, name), dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_training_diagnostics(train_output: Dict[str, object], save_dir: Optional[str] = None, show: bool = True):
    losses = train_output["losses"]
    fig = plt.figure(figsize=(8, 4.5))
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("policy imitation epoch")
    plt.ylabel("MSE loss")
    plt.title("Deep GPS policy learns local LQR-guided residual teachers")
    plt.grid(True, linestyle="--", alpha=0.6)
    _save_or_show(fig, save_dir, "policy_imitation_loss.png", show)

    fig = plt.figure(figsize=(9, 5))
    for res in train_output["local_results"]:
        hist = res["history"]
        plt.plot([h["iteration"] for h in hist], [h["best"] for h in hist], alpha=0.75)
    plt.xlabel("CEM iteration")
    plt.ylabel("best local objective")
    plt.title("Local GPS teacher optimization progress")
    plt.grid(True, linestyle="--", alpha=0.6)
    _save_or_show(fig, save_dir, "local_gps_teacher_progress.png", show)


def plot_evaluation_summary(rows: Sequence[Dict[str, object]], save_dir: Optional[str] = None, show: bool = True):
    labels = [f"{r['case'].theta_goal_deg:.0f}°\n{r['case'].alpha_deg:.0f}/{r['case'].phi_deg:.0f}°" for r in rows]
    x = np.arange(len(rows))
    width = 0.25
    c_existing = np.array([r["existing_metrics"]["total"] for r in rows])
    c_nn = np.array([r["nn_metrics"]["total"] for r in rows])
    c_gps = np.array([r["gps_metrics"]["total"] for r in rows])

    fig = plt.figure(figsize=(max(10, len(rows) * 0.55), 5))
    plt.bar(x - width, c_existing, width, label="existing LQR reference")
    plt.bar(x, c_nn, width, label="deep policy only")
    plt.bar(x + width, c_gps, width, label="true GPS refined")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("objective cost")
    plt.title("Trajectory objective comparison")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    _save_or_show(fig, save_dir, "summary_objective_cost.png", show)

    improvement = 100.0 * (c_existing - c_gps) / np.maximum(np.abs(c_existing), 1e-12)
    fig = plt.figure(figsize=(max(10, len(rows) * 0.55), 5))
    plt.bar(x, improvement, label="GPS improvement over existing")
    plt.axhline(0.0, linestyle=":")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("improvement [%]")
    plt.title("Positive means the GPS trajectory improves the base LQR trajectory")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    _save_or_show(fig, save_dir, "summary_improvement_percent.png", show)

    def bar_metric(metric: str, ylabel: str, title: str, limit: Optional[float] = None, filename: str = "metric.png"):
        y_e = np.array([r["existing_metrics"][metric] for r in rows])
        y_n = np.array([r["nn_metrics"][metric] for r in rows])
        y_g = np.array([r["gps_metrics"][metric] for r in rows])
        fig = plt.figure(figsize=(max(10, len(rows) * 0.55), 5))
        plt.bar(x - width, y_e, width, label="existing")
        plt.bar(x, y_n, width, label="deep policy")
        plt.bar(x + width, y_g, width, label="GPS refined")
        if limit is not None:
            plt.axhline(limit, linestyle=":", label="limit")
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, axis="y", linestyle="--", alpha=0.6)
        _save_or_show(fig, save_dir, filename, show)

    bar_metric("energy", "normalized command energy", "Controller command energy", None, "summary_energy.png")
    bar_metric("max_abs_omega", "rad/s", "Max actual |theta_dot|", 0.2, "summary_max_omega.png")
    bar_metric("max_abs_tau_m", "N.m", "Max shaft torque |tau_m|", 2000.0, "summary_max_tau_m.png")
    bar_metric("final_theta_error", "rad", "Final theta error", None, "summary_final_error.png")


def plot_case_comparison(
    row: Dict[str, object],
    obj_cfg: GPSObjectiveConfig,
    save_dir: Optional[str] = None,
    show: bool = True,
):
    case: GPSCase = row["case"]
    logs_e = row["existing_logs"]
    logs_n = row["nn_logs"]
    logs_g = row["gps_logs"]
    t_e = np.asarray(logs_e["t"])
    t_n = np.asarray(logs_n["t"])
    t_g = np.asarray(logs_g["t"])

    fig, axes = plt.subplots(6, 1, figsize=(11, 17), sharex=False)
    fig.suptitle(f"True GPS guided trajectory vs existing LQR\n{case.label()}")

    axes[0].plot(t_e, logs_e["theta_ref"], "--", label="existing theta_ref")
    axes[0].plot(t_e, logs_e["theta"], label="existing theta")
    axes[0].plot(t_n, logs_n["theta_ref"], "--", label="NN theta_ref")
    axes[0].plot(t_n, logs_n["theta"], label="NN theta")
    axes[0].plot(t_g, logs_g["theta_ref"], "--", label="GPS theta_ref")
    axes[0].plot(t_g, logs_g["theta"], label="GPS theta")
    axes[0].axhline(case.theta_goal, linestyle=":", label="theta_goal")
    axes[0].set_ylabel("theta [rad]")
    axes[0].legend(loc="best")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(t_e, np.asarray(logs_e["theta"]) - np.asarray(logs_e["theta_ref"]), label="existing theta error")
    axes[1].plot(t_n, np.asarray(logs_n["theta"]) - np.asarray(logs_n["theta_ref"]), label="NN theta error")
    axes[1].plot(t_g, np.asarray(logs_g["theta"]) - np.asarray(logs_g["theta_ref"]), label="GPS theta error")
    axes[1].set_ylabel("theta error [rad]")
    axes[1].legend(loc="best")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    axes[2].plot(t_e, logs_e["omega"], label="existing omega")
    axes[2].plot(t_e, logs_e["omega_ref"], "--", label="existing omega_ref")
    axes[2].plot(t_g, logs_g["omega"], label="GPS omega")
    axes[2].plot(t_g, logs_g["omega_ref"], "--", label="GPS omega_ref")
    axes[2].axhline(obj_cfg.omega_limit, linestyle=":", label="+omega limit")
    axes[2].axhline(-obj_cfg.omega_limit, linestyle=":", label="-omega limit")
    axes[2].set_ylabel("theta_dot [rad/s]")
    axes[2].legend(loc="best")
    axes[2].grid(True, linestyle="--", alpha=0.6)

    axes[3].plot(t_e, logs_e["u_total"], label="existing u_total")
    axes[3].plot(t_n, logs_n["u_total"], label="NN u_total")
    axes[3].plot(t_g, logs_g["u_total"], label="GPS u_total")
    axes[3].axhline(obj_cfg.command_limit, linestyle=":", label="+command limit")
    axes[3].axhline(-obj_cfg.command_limit, linestyle=":", label="-command limit")
    axes[3].set_ylabel("controller command")
    axes[3].legend(loc="best")
    axes[3].grid(True, linestyle="--", alpha=0.6)

    tau_e = np.asarray(logs_e.get("tau_m", logs_e["u_total"]))
    tau_n = np.asarray(logs_n.get("tau_m", logs_n["u_total"]))
    tau_g = np.asarray(logs_g.get("tau_m", logs_g["u_total"]))
    axes[4].plot(t_e, tau_e, label="existing tau_m")
    axes[4].plot(t_n, tau_n, label="NN tau_m")
    axes[4].plot(t_g, tau_g, label="GPS tau_m")
    axes[4].axhline(obj_cfg.torque_limit, linestyle=":", label="+torque limit")
    axes[4].axhline(-obj_cfg.torque_limit, linestyle=":", label="-torque limit")
    axes[4].set_ylabel("shaft torque [N.m]")
    axes[4].legend(loc="best")
    axes[4].grid(True, linestyle="--", alpha=0.6)

    # Show the residual nature of the GPS trajectory on normalized time.
    z_e = np.linspace(0.0, 1.0, len(row["base_ref"]))
    z_g = np.linspace(0.0, 1.0, len(row["gps_ref"]))
    base_on_g = np.interp(z_g, z_e, row["base_ref"])
    axes[5].plot(z_g, base_on_g, label="LQR guide shape")
    axes[5].plot(z_g, row["gps_ref"], label="GPS learned shape")
    axes[5].plot(z_g, np.asarray(row["gps_ref"]) - base_on_g, label="learned correction Δtheta")
    axes[5].set_xlabel("normalized time z")
    axes[5].set_ylabel("trajectory / correction [rad]")
    axes[5].legend(loc="best")
    axes[5].grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fname = f"case_true_gps_goal_{case.theta_goal_deg:.0f}_tilt_{case.alpha_deg:.0f}_{case.phi_deg:.0f}.png"
    _save_or_show(fig, save_dir, fname, show)


def save_evaluation_table(rows: Sequence[Dict[str, object]], save_dir: str):
    mkdir(save_dir)
    table = []
    for r in rows:
        c = r["case"]
        table.append(dict(
            theta_goal_deg=c.theta_goal_deg,
            alpha_deg=c.alpha_deg,
            phi_deg=c.phi_deg,
            existing_cost=r["existing_metrics"]["total"],
            nn_cost=r["nn_metrics"]["total"],
            gps_cost=r["gps_metrics"]["total"],
            gps_improvement_percent=100.0 * (r["existing_metrics"]["total"] - r["gps_metrics"]["total"]) / max(abs(r["existing_metrics"]["total"]), 1e-12),
            existing_energy=r["existing_metrics"]["energy"],
            gps_energy=r["gps_metrics"]["energy"],
            existing_max_omega=r["existing_metrics"]["max_abs_omega"],
            gps_max_omega=r["gps_metrics"]["max_abs_omega"],
            existing_max_tau_m=r["existing_metrics"]["max_abs_tau_m"],
            gps_max_tau_m=r["gps_metrics"]["max_abs_tau_m"],
            existing_final_error=r["existing_metrics"]["final_theta_error"],
            gps_final_error=r["gps_metrics"]["final_theta_error"],
        ))
    with open(os.path.join(save_dir, "true_gps_evaluation_summary.json"), "w") as f:
        json.dump(table, f, indent=2)


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

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
) -> Dict[str, object]:
    """Train and evaluate true GPS trajectory policy.

    Increase cem_iters/population/policy_epochs for final-quality results.
    Use small values for a smoke test.
    """
    mkdir(save_dir)
    traj_cfg = GPSTrajectoryConfig(
        n_basis=6,
        omega_ref_limit=0.2,
        duration_margin=1.05,
        max_duration_factor=2.5,
        min_extra_duration=5.0,
        hidden_sizes=(96, 96),
    )
    obj_cfg = GPSObjectiveConfig(
        omega_limit=0.2,
        torque_limit=2000.0,
        command_limit=500.0,
    )

    train_cases = make_cases(train_goal_degs, tilt_degs, coupled_tilts=coupled_tilts)
    train_output = train_true_gps_policy(
        train_cases=train_cases,
        traj_cfg=traj_cfg,
        obj_cfg=obj_cfg,
        cem_iters=cem_iters,
        population=population,
        elite_frac=0.20,
        policy_epochs=policy_epochs,
        seed=1,
        save_dir=save_dir,
    )

    test_cases = make_cases(test_goal_degs, tilt_degs, coupled_tilts=coupled_tilts)
    rows = evaluate_true_gps_policy(
        policy=train_output["policy"],
        test_cases=test_cases,
        traj_cfg=traj_cfg,
        obj_cfg=obj_cfg,
        refinement_iters=refinement_iters,
        refinement_population=refinement_population,
        seed=1000,
    )

    plot_training_diagnostics(train_output, save_dir=save_dir, show=show_plots)
    plot_evaluation_summary(rows, save_dir=save_dir, show=show_plots)

    # Plot representative cases: low/high tilt and middle/high goals.
    for row in rows:
        g = round(row["case"].theta_goal_deg)
        a = round(row["case"].alpha_deg)
        if (g in (75, 105, 180)) and (a in (0, 20)):
            plot_case_comparison(row, obj_cfg, save_dir=save_dir, show=show_plots)

    save_evaluation_table(rows, save_dir)

    n_improved = sum(1 for r in rows if r["gps_metrics"]["total"] < r["existing_metrics"]["total"])
    print("\n=== TRUE GPS SUMMARY ===")
    print(f"GPS-refined trajectory improved {n_improved}/{len(rows)} test cases.")
    print(f"Results saved in: {os.path.abspath(save_dir)}")
    print("Primary success criterion: gps_cost < existing_cost in true_gps_evaluation_summary.json")

    return dict(train_output=train_output, evaluation_rows=rows, traj_cfg=traj_cfg, obj_cfg=obj_cfg)


#%% Default here is a smoke test. For final-quality runs, call run_true_gps_experiment
# from another script with larger cem_iters/population/policy_epochs.
if __name__ == "__main__":
    run_true_gps_experiment(
        train_goal_degs=(30, 60, 90, 120, 150, 180),
        test_goal_degs=(15, 45, 75, 105, 135, 165, 180),
        tilt_degs=(0, 20),
        cem_iters=30,
        population=50,
        policy_epochs=3000,
        refinement_iters=2,
        refinement_population=10,
        save_dir="true_gps_results_smoke",
        show_plots=True,
    )
