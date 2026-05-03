# -*- coding: utf-8 -*-
"""
Single-file trajectory evaluation for the 1-DOF rotor.

Run this file cell-by-cell in Spyder.
Edit only the USER SETTINGS cell for normal experiments.

trj_type chooses only the reference trajectory theta_ref(t):
    "LQR" : finite-horizon LQR reference from this file
    "NN"  : load trajectory_rl_gps_addon.py saved trajectory_policy.pt
    "RL"  : load sac_gps_lqr_guided_saved_teacher_addon.py saved SAC-GPS actor

No NN/RL training is done here.  The controller remains pure TDE+SMC.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import os
import math
import numpy as np


#%% ========================= USER SETTINGS =========================
# Edit ONLY this cell for normal experiments.
# Put every user-changeable number/path here.

TRJ_TYPE = "LQR"          # "LQR", "NN", or "RL"
THETA0_DEG = 0.0
THETA_GOAL_DEG = -90.0
SEED = 0

# LQR duration override.  This keeps your old behavior where you used
# reference={"duration": 150.2}.  Set None for automatic 4 sec per pi rad.
LQR_DURATION_S = 150.2

# Saved-weight folders/files.
NN_WEIGHTS_DIR = "trajectory_rl_results"
NN_WEIGHTS_FILE = "trajectory_policy.pt"
RL_WEIGHTS_DIR = "true_gps_results_smoke"
RL_CHECKPOINT_FILE = "sac_gps_agent.pt"
RL_ACTOR_FILE = "sac_gps_actor.pt"

# Plant / actuator / geometry.
PLANT_J = 14099.0
PLANT_B = 0.09
U_MAX = 500.0
OMEGA_MAX = 8.0
DT = 0.002
TAU_I = 0.1
K_T = 5.0
M1 = 5000.0
R1 = 3.10
M2 = 600.5
A2 = 1.5
R_BLUE = 0.5
M3 = 600.8
A3 = 1.5
R_ORANGE = 0.5
M4 = 300.4
L4 = 1.20
L4_C = 0.60
GAMMA = 0.0
BETA2 = 0.0
BETA4 = 0.0
ALPHA_DEG = 5.0
PHI_DEG = 3.0
GRAVITY = 9.81
SUBTRACT_GRAVITY_IN_UEQ = False

# Nominal model used by LQR and TDE+SMC.
NOMINAL_J = 13000.0
NOMINAL_B = 0.06

# LQR trajectory weights.
LQR_Q_THETA = 85.0
LQR_Q_OMEGA = 18.0
LQR_R_U = 0.02
LQR_QT_THETA = 4200.0
LQR_QT_OMEGA = 220.0
LQR_OMEGA_LIMIT_PENALTY = 1000.0

# TDE+SMC controller gains.
SMC_LAMBDA = 35.0
SMC_K = 0.85
SMC_PHI = 0.025

# Rollout cost used only for reporting total_cost.
COST_W_E = 8.0
COST_W_EDOT = 1.0
COST_W_U = 0.03
COST_W_OMEGA = 0.3
GOAL_TOL = 1e-2
DONE_BONUS = 2.0

# NN trajectory decoder; must match trajectory_rl_gps_addon.py training config.
NN_N_SHAPE = 5
NN_HIDDEN_SIZES = (64, 64)
NN_OMEGA_REF_LIMIT = 0.2
NN_MIN_DURATION = 0.5
NN_MAX_DURATION = 80.0
NN_MAX_TILT_DEG = 20.0

# SAC-GPS/RL actor and LQR-guided residual decoder; must match SAC-GPS config.
RL_N_BASIS = 6
RL_HIDDEN_SIZES = (128, 128)
RL_SHAPE_ACTION_SCALE = 1.30
RL_DURATION_ACTION_SCALE = 1.20
RL_OMEGA_REF_LIMIT = 0.2
RL_DURATION_MARGIN = 1.05
RL_MAX_DURATION_FACTOR = 2.5
RL_MIN_EXTRA_DURATION = 5.0
RL_DURATION_EXP_SCALE = 0.35
RL_Z_GRID_SIZE = 1201
RL_MAX_TILT_DEG = 20.0


#%% ========================= DATA CONTAINERS =========================

@dataclass
class PlantParams:
    J: float
    b: float
    u_max: float
    omega_max: float
    dt: float
    tau_i: float
    K_t: float
    m1: float
    R1: float
    m2: float
    a2: float
    r1: float
    m3: float
    a3: float
    r2: float
    m4: float
    L4: float
    l4_c: float
    gamma: float
    beta2: float
    beta4: float
    alpha: float
    phi: float
    g: float
    subtract_gravity_in_ueq: bool

@dataclass
class NominalModel:
    J: float
    b: float

@dataclass
class LQRWeights:
    q_theta: float
    q_omega: float
    r_u: float
    qT_theta: float
    qT_omega: float
    omega_limit_penalty: float

@dataclass
class SMCConfig:
    lambda_s: float
    k: float
    phi: float

@dataclass
class CostConfig:
    w_e: float
    w_edot: float
    w_u: float
    w_omega: float
    goal_tol: float
    done_bonus: float

@dataclass
class Task:
    theta0: float
    omega0: float
    theta_goal: float


#%% ========================= BASIC UTILITIES =========================

def sat(x: float, limit: float) -> float:
    return max(-limit, min(limit, float(x)))

def finite_diff(y: np.ndarray, dt: float) -> np.ndarray:
    dy = np.zeros_like(y, dtype=float)
    if len(y) >= 2:
        dy[0] = (y[1] - y[0]) / dt
        dy[-1] = (y[-1] - y[-2]) / dt
    if len(y) >= 3:
        dy[1:-1] = (y[2:] - y[:-2]) / (2.0 * dt)
    return dy

def trapz_safe(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(y) < 2:
        return 0.0
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))

def cumtrapz_safe(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(y, dtype=float)
    if len(y) >= 2:
        out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * np.diff(x))
    return out

def minimum_jerk(z: np.ndarray) -> np.ndarray:
    return 10.0 * z**3 - 15.0 * z**4 + 6.0 * z**5

def sin_basis(z: np.ndarray, n: int) -> np.ndarray:
    return np.stack([np.sin(k * math.pi * z) for k in range(1, n + 1)], axis=1)

def softplus_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


#%% ========================= PLANT DYNAMICS =========================

def J_local_cylinder_z(m: float, a: float) -> float:
    return 0.5 * m * a * a

def J_bar_inplane_COM(m: float, L: float) -> float:
    return (1.0 / 12.0) * m * L * L

def red_link_radius_from_pin(r2: float, l_c: float, gamma: float) -> float:
    return math.sqrt(max(0.0, r2 * r2 + l_c * l_c + 2.0 * r2 * l_c * math.cos(gamma)))

def compute_equivalent_inertia(p: PlantParams) -> float:
    J1 = 0.5 * p.m1 * p.R1 * p.R1 if p.m1 > 0 and p.R1 > 0 else 0.0
    J2_local = J_local_cylinder_z(p.m2, p.a2) if p.m2 > 0 else 0.0
    J3_local = J_local_cylinder_z(p.m3, p.a3) if p.m3 > 0 else 0.0
    R4 = red_link_radius_from_pin(p.r2, p.l4_c, p.gamma) if p.m4 > 0 else 0.0
    J4_com = J_bar_inplane_COM(p.m4, p.L4) if p.m4 > 0 else 0.0
    J2 = J2_local + p.m2 * p.r1 * p.r1
    J3 = J3_local + p.m3 * p.r2 * p.r2
    J4 = J4_com + p.m4 * R4 * R4
    return J1 + J2 + J3 + J4

def gravity_proj_inplane(alpha: float, phi: float, g: float) -> Tuple[float, float]:
    gx = -g * math.sin(phi)
    gy = g * math.sin(alpha) * math.cos(phi)
    psi = math.atan2(gy, gx)
    g_t = math.hypot(gx, gy)
    return g_t, psi

class OneDOFRotorPlant:
    def __init__(self, p: PlantParams):
        self.p = p
        self.state = np.zeros(3)  # [theta, omega, tau_m]
        self.J_eq = compute_equivalent_inertia(self.p)
        self.torque_coulomb = 0.02
        self.load_amp = 0.03
        self.load_freq = 1.5
        self.noise_std = 0.01
        self.extra_base_amp = 0.02
        self.extra_base_freq = 2.0
        self.extra_amp_mod = 0.5
        self.extra_amp_mod_freq = 0.1
        self.extra_freq_mod = 0.5
        self.extra_freq_mod_freq = 0.05
        self.time = 0.0
        self.last_disturbance = 0.0
        self.last_tau_m = 0.0
        self.last_command = 0.0
        self.last_gravity = 0.0

    def gravity_torque(self, theta: float) -> float:
        g_t, psi = gravity_proj_inplane(self.p.alpha, self.p.phi, self.p.g)
        R4 = red_link_radius_from_pin(self.p.r2, self.p.l4_c, self.p.gamma) if self.p.m4 > 0 else 0.0
        tau2 = self.p.m2 * self.p.r1 * g_t * math.sin(theta + self.p.beta2 - psi) if self.p.m2 > 0 else 0.0
        tau4 = self.p.m4 * R4 * g_t * math.sin(theta + self.p.beta4 - psi) if self.p.m4 > 0 else 0.0
        return tau2 + tau4

    def reset(self, theta0: float = 0.0, omega0: float = 0.0, tau_m0: float = 0.0) -> np.ndarray:
        self.state[:] = [theta0, omega0, tau_m0]
        self.J_eq = compute_equivalent_inertia(self.p)
        self.time = 0.0
        self.last_disturbance = 0.0
        self.last_tau_m = tau_m0
        self.last_command = 0.0
        self.last_gravity = 0.0
        return self.state.copy()

    def step(self, u_cmd: float) -> np.ndarray:
        u = sat(u_cmd, self.p.u_max)
        theta, omega, tau_m = self.state
        d_coul = self.torque_coulomb * (1.0 if omega >= 0 else -1.0) if abs(omega) > 1e-5 else 0.0
        d_per = self.load_amp * math.sin(2.0 * math.pi * self.load_freq * self.time)
        amp_var = self.extra_base_amp * (1.0 + self.extra_amp_mod * math.sin(2.0 * math.pi * self.extra_amp_mod_freq * self.time))
        freq_var = self.extra_base_freq * (1.0 + self.extra_freq_mod * math.sin(2.0 * math.pi * self.extra_freq_mod_freq * self.time))
        d_var = amp_var * math.sin(2.0 * math.pi * freq_var * self.time)
        d_noise = np.random.randn() * self.noise_std
        tau_g = self.gravity_torque(theta)
        d = d_coul + d_per + d_var + d_noise + tau_g
        self.last_disturbance = d
        self.last_tau_m = tau_m
        self.last_command = u
        self.last_gravity = tau_g

        domega = (tau_m - self.p.b * omega + d) / self.J_eq
        omega_next = omega + domega * self.p.dt
        theta_next = theta + omega_next * self.p.dt
        tau_m_next = tau_m + self.p.dt * (-(1.0 / self.p.tau_i) * tau_m + (self.p.K_t / self.p.tau_i) * u)
        self.state[:] = [theta_next, omega_next, tau_m_next]
        self.time += self.p.dt
        return self.state.copy()


#%% ========================= LQR REFERENCE =========================

def build_AB(nom: NominalModel, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    A = np.array([[1.0, dt], [0.0, 1.0 - (nom.b / nom.J) * dt]])
    B = np.array([[0.0], [dt / nom.J]])
    return A, B

def finite_horizon_lqr_gain(A: np.ndarray, B: np.ndarray, N: int, w: LQRWeights) -> List[np.ndarray]:
    Q = np.diag([w.q_theta, w.q_omega])
    R = np.array([[w.r_u]])
    Qf = np.diag([w.qT_theta, w.qT_omega])
    P = Qf.copy()
    Ks: List[np.ndarray] = [np.zeros((1, 2)) for _ in range(N)]
    for k in reversed(range(N)):
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        Ks[k] = K
        P = Q + A.T @ P @ (A - B @ K)
    return Ks

def generate_reference_ilqr_like(nom: NominalModel, plant_limits: PlantParams, x0: np.ndarray, x_goal: np.ndarray, N: int, dt: float, w: LQRWeights) -> Tuple[np.ndarray, np.ndarray]:
    A, B = build_AB(nom, dt)
    Ks = finite_horizon_lqr_gain(A, B, N, w)
    x_ref = np.zeros((N + 1, 2))
    u_ref = np.zeros(N)
    x = x0.copy()
    for k in range(N):
        x_tilde = x - x_goal
        excess = abs(x[1]) - plant_limits.omega_max
        if excess > 0:
            x_tilde[1] += w.omega_limit_penalty * excess * math.copysign(1.0, x[1])
        u = float((-Ks[k] @ x_tilde.reshape(2, 1)).item())
        u = sat(u, plant_limits.u_max)
        x = A @ x + B.flatten() * u
        x_ref[k, :] = x
        u_ref[k] = u
    x_ref[N, :] = x
    return x_ref, u_ref

def time_horizon(theta0: float, theta_goal: float) -> float:
    return 4.0 * abs(theta_goal - theta0) / math.pi if abs(theta_goal - theta0) > 1e-12 else 0.5

def explicit_lqr_reference(theta0: float, theta_goal: float, plant_p: PlantParams, nom: NominalModel, lqr_w: LQRWeights, duration: Optional[float] = None) -> Tuple[np.ndarray, float]:
    T = time_horizon(theta0, theta_goal) if duration is None else float(duration)
    N = max(1, int(round(T / plant_p.dt)))
    T = N * plant_p.dt
    x_ref, _ = generate_reference_ilqr_like(
        nom, plant_p,
        np.array([theta0, 0.0], dtype=float),
        np.array([theta_goal, 0.0], dtype=float),
        N, plant_p.dt, lqr_w,
    )
    theta_ref = np.concatenate([[theta0], x_ref[:-1, 0]])
    return theta_ref, T


#%% ========================= TDE + SMC CONTROLLER =========================

class TDE_SMC_Discrete:
    def __init__(self, hatJ: float, hatb: float, dt: float, u_max: float, tau_i: float, K_t: float, smc: SMCConfig):
        self.hatJ = hatJ
        self.hatb = hatb
        self.dt = dt
        self.u_max = u_max
        self.tau_i = tau_i
        self.K_t = K_t
        self.cfg = smc
        self.tau_m_hat = 0.0
        self.omega_hist: List[float] = []
        self.tau_m_hat_hist: List[float] = []
        self.subtract_gravity_in_ueq = False

    def reset(self):
        self.tau_m_hat = 0.0
        self.omega_hist.clear()
        self.tau_m_hat_hist.clear()

    def control(self, theta: float, omega: float, theta_ref_k: float, theta_ref_k1: float, omega_ref_k: float, omega_ref_k1: float, u_rl: float = 0.0, plant: Optional[OneDOFRotorPlant] = None) -> Tuple[float, Dict[str, float]]:
        e = theta - theta_ref_k
        edot = omega - omega_ref_k
        s = edot + self.cfg.lambda_s * e

        self.omega_hist.append(omega)
        if len(self.omega_hist) > 2:
            self.omega_hist.pop(0)

        eta_hat = 0.0
        d_hat = 0.0
        if self.tau_m_hat_hist:
            eta_hat = self.tau_m_hat_hist[-1] - self.hatJ * self.omega_hist[-1]
            d_hat = eta_hat - self.hatb * omega

        dtheta_ref = theta_ref_k1 - theta_ref_k
        domega_ref = omega_ref_k1 - omega_ref_k
        denom = self.dt * (1.0 + self.cfg.lambda_s * self.dt)
        u_eq = self.hatb * omega + (self.hatJ / denom) * (domega_ref - self.cfg.lambda_s * (self.dt * omega - dtheta_ref))
        if self.subtract_gravity_in_ueq and plant is not None:
            u_eq = u_eq - plant.gravity_torque(theta)

        u_s = -self.cfg.k * np.tanh(s / (self.cfg.phi + 1e-9))
        u_total = sat(u_eq - d_hat + u_s + u_rl, self.u_max)

        tau_m_hat_next = self.tau_m_hat + self.dt * (-(1.0 / self.tau_i) * self.tau_m_hat + (self.K_t / self.tau_i) * u_total)
        self.tau_m_hat = tau_m_hat_next
        self.tau_m_hat_hist.append(tau_m_hat_next)
        if len(self.tau_m_hat_hist) > 2:
            self.tau_m_hat_hist.pop(0)

        info = dict(e=e, edot=edot, s=s, eta_hat=eta_hat, d_hat=d_hat, u_eq=u_eq, u_s=u_s, u_smc=u_eq - d_hat + u_s, u_total=u_total)
        return u_total, info


#%% ========================= SYSTEM BUILDER =========================

def build_system_from_settings():
    plant_p = PlantParams(
        J=PLANT_J, b=PLANT_B, u_max=U_MAX, omega_max=OMEGA_MAX, dt=DT,
        tau_i=TAU_I, K_t=K_T, m1=M1, R1=R1, m2=M2, a2=A2, r1=R_BLUE,
        m3=M3, a3=A3, r2=R_ORANGE, m4=M4, L4=L4, l4_c=L4_C,
        gamma=GAMMA, beta2=BETA2, beta4=BETA4, alpha=math.radians(ALPHA_DEG),
        phi=math.radians(PHI_DEG), g=GRAVITY, subtract_gravity_in_ueq=SUBTRACT_GRAVITY_IN_UEQ,
    )
    nom = NominalModel(J=NOMINAL_J, b=NOMINAL_B)
    lqr_w = LQRWeights(q_theta=LQR_Q_THETA, q_omega=LQR_Q_OMEGA, r_u=LQR_R_U, qT_theta=LQR_QT_THETA, qT_omega=LQR_QT_OMEGA, omega_limit_penalty=LQR_OMEGA_LIMIT_PENALTY)
    smc_cfg = SMCConfig(lambda_s=SMC_LAMBDA, k=SMC_K, phi=SMC_PHI)
    cost_cfg = CostConfig(w_e=COST_W_E, w_edot=COST_W_EDOT, w_u=COST_W_U, w_omega=COST_W_OMEGA, goal_tol=GOAL_TOL, done_bonus=DONE_BONUS)
    return plant_p, nom, lqr_w, smc_cfg, cost_cfg


#%% ========================= NN / RL TRAJECTORY LOADING =========================

def torch_import():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError(f"PyTorch is required for trj_type='NN' or 'RL': {exc}")
    return torch, nn

def case_features(theta_goal: float, alpha: float, phi: float, max_tilt_deg: float) -> np.ndarray:
    max_tilt = math.radians(max_tilt_deg)
    return np.array([theta_goal / math.pi, alpha / max_tilt, phi / max_tilt], dtype=np.float32)

def decode_nn_reference(theta0: float, theta_goal: float, params: np.ndarray, dt: float) -> Tuple[np.ndarray, float]:
    params = np.asarray(params, dtype=float).reshape(-1)
    if params.size != NN_N_SHAPE + 1:
        raise ValueError(f"NN params length must be {NN_N_SHAPE + 1}, got {params.size}")
    delta = theta_goal - theta0
    if abs(delta) < 1e-12:
        t = np.arange(0.0, NN_MIN_DURATION + 0.5 * dt, dt)
        return np.full_like(t, theta0), float(t[-1])

    z = np.linspace(0.0, 1.0, 1001)
    logits = sin_basis(z, NN_N_SHAPE) @ params[:NN_N_SHAPE]
    v = z * (1.0 - z) * np.exp(np.clip(logits, -5.0, 5.0)) + 1e-6
    v /= max(trapz_safe(v, z), 1e-12)
    h = cumtrapz_safe(v, z)
    h /= max(h[-1], 1e-12)
    T_min = abs(delta) * float(np.max(np.abs(v))) / max(NN_OMEGA_REF_LIMIT, 1e-12)
    T = float(np.clip(T_min + float(softplus_np([params[-1]])[0]), NN_MIN_DURATION, NN_MAX_DURATION))
    N = max(1, int(round(T / dt)))
    t = np.arange(N + 1) * dt
    T = float(t[-1])
    theta_ref = theta0 + delta * np.interp(t / max(T, 1e-12), z, h)
    theta_ref[0] = theta0
    theta_ref[-1] = theta_goal
    return theta_ref, T

def load_nn_reference(theta0: float, theta_goal: float, plant_p: PlantParams) -> Tuple[np.ndarray, float]:
    """Load trajectory_rl_gps_addon.py saved trajectory_policy.pt without importing that script."""
    torch, nn = torch_import()
    path = os.path.join(NN_WEIGHTS_DIR, NN_WEIGHTS_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"NN weights not found: {path}")

    h1, h2 = NN_HIDDEN_SIZES
    net = nn.Sequential(
        nn.Linear(3, h1), nn.Tanh(),
        nn.Linear(h1, h2), nn.Tanh(),
        nn.Linear(h2, NN_N_SHAPE + 1),
    )
    state = torch.load(path, map_location="cpu")
    # TrajectoryPolicyNet has self.net = nn.Sequential(...), so saved keys are net.0.weight, ...
    # This file uses a bare Sequential, so strip the leading "net." prefix.
    if isinstance(state, dict) and any(str(k).startswith("net.") for k in state.keys()):
        state = {str(k).replace("net.", "", 1): v for k, v in state.items() if str(k).startswith("net.")}
    net.load_state_dict(state)
    net.eval()

    x = torch.as_tensor(case_features(theta_goal, plant_p.alpha, plant_p.phi, NN_MAX_TILT_DEG), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        params = net(x).squeeze(0).cpu().numpy().astype(float)
    params[:-1] = np.clip(params[:-1], -4.0, 4.0)
    params[-1] = np.clip(params[-1], -4.0, 12.0)
    return decode_nn_reference(theta0, theta_goal, params, plant_p.dt)

def lqr_guide_shape(theta_ref: np.ndarray, theta_goal: float, n_grid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_old = np.linspace(0.0, 1.0, len(theta_ref))
    z = np.linspace(0.0, 1.0, n_grid)
    raw = np.asarray(theta_ref, dtype=float) - float(theta_ref[0])
    scale = float(raw[-1]) if len(raw) else 0.0
    if abs(theta_goal) < 1e-12:
        h = np.zeros_like(z)
    elif abs(scale) < 1e-9:
        h = minimum_jerk(z)
    else:
        h_old = np.maximum.accumulate(np.clip(raw / scale, 0.0, 1.0))
        h_old[0] = 0.0
        h_old[-1] = 1.0
        h = np.interp(z, z_old, h_old)
        h = np.maximum.accumulate(np.clip(0.85 * h + 0.15 * minimum_jerk(z), 0.0, 1.0))
        h[0] = 0.0
        h[-1] = 1.0
    v = np.maximum(np.gradient(h, z), 1e-6)
    v = np.minimum(v, max(5.0, 3.0 * float(np.percentile(v, 95))))
    v /= max(trapz_safe(v, z), 1e-12)
    return z, h, v

def decode_rl_duration(raw_duration: float, theta_goal: float, T_guide: float) -> float:
    delta = abs(theta_goal)
    if delta < 1e-12:
        return max(0.5, T_guide)
    T_min = RL_DURATION_MARGIN * delta / RL_OMEGA_REF_LIMIT
    T_center = max(T_guide, T_min)
    T_max = max(RL_MAX_DURATION_FACTOR * T_min, T_min + RL_MIN_EXTRA_DURATION, T_center)
    return float(np.clip(T_center * math.exp(RL_DURATION_EXP_SCALE * float(raw_duration)), T_min, T_max))

def decode_rl_reference(theta0: float, theta_goal: float, params: np.ndarray, base_ref: np.ndarray, base_T: float, dt: float) -> Tuple[np.ndarray, float]:
    params = np.asarray(params, dtype=float).reshape(-1)
    if params.size != RL_N_BASIS + 1:
        raise ValueError(f"RL params length must be {RL_N_BASIS + 1}, got {params.size}")
    delta = theta_goal - theta0
    if abs(delta) < 1e-12:
        N = max(1, int(math.ceil(max(0.5, base_T) / dt)))
        return np.full(N + 1, theta0), N * dt

    z, _, v_base = lqr_guide_shape(base_ref, theta_goal, RL_Z_GRID_SIZE)
    residual_log = np.clip(sin_basis(z, RL_N_BASIS) @ params[:RL_N_BASIS], -3.0, 3.0)
    v = np.maximum(v_base, 1e-8) * np.exp(residual_log)
    v = np.maximum(v, 1e-9)
    v /= max(trapz_safe(v, z), 1e-12)

    T = decode_rl_duration(params[-1], theta_goal, base_T)
    T_needed = RL_DURATION_MARGIN * abs(delta) * float(np.max(v)) / RL_OMEGA_REF_LIMIT
    T = max(T, T_needed)
    N = max(1, int(math.ceil(T / dt)))
    T = N * dt

    h = cumtrapz_safe(v, z)
    h /= max(h[-1], 1e-12)
    h = np.maximum.accumulate(np.clip(h, 0.0, 1.0))
    h[0] = 0.0
    h[-1] = 1.0
    t = np.arange(N + 1) * dt
    theta_ref = theta0 + delta * np.interp(t / max(T, 1e-12), z, h)
    theta_ref[0] = theta0
    theta_ref[-1] = theta_goal
    return theta_ref, T

def load_rl_reference(theta0: float, theta_goal: float, plant_p: PlantParams, nom: NominalModel, lqr_w: LQRWeights) -> Tuple[np.ndarray, float]:
    """Load SAC-GPS actor without importing sac_gps_lqr_guided_saved_teacher_addon.py."""
    torch, nn = torch_import()
    p_agent = os.path.join(RL_WEIGHTS_DIR, RL_CHECKPOINT_FILE)
    p_actor = os.path.join(RL_WEIGHTS_DIR, RL_ACTOR_FILE)
    path = p_agent if os.path.exists(p_agent) else p_actor
    if not os.path.exists(path):
        raise FileNotFoundError(f"RL actor weights not found in {RL_WEIGHTS_DIR}. Expected {RL_CHECKPOINT_FILE} or {RL_ACTOR_FILE}.")

    act_dim = RL_N_BASIS + 1
    h1, h2 = RL_HIDDEN_SIZES
    net = nn.Sequential(
        nn.Linear(3, h1), nn.ReLU(),
        nn.Linear(h1, h2), nn.ReLU(),
        nn.Linear(h2, 2 * act_dim),
    )
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("actor", ckpt) if isinstance(ckpt, dict) else ckpt
    # SquashedGaussianActor has self.net = MLP; MLP has self.net = Sequential.
    # So saved keys are net.net.0.weight, ... .  Strip net.net. for bare Sequential.
    if isinstance(state, dict) and any(str(k).startswith("net.net.") for k in state.keys()):
        state = {str(k).replace("net.net.", "", 1): v for k, v in state.items() if str(k).startswith("net.net.")}
    net.load_state_dict(state)
    net.eval()

    obs = case_features(theta_goal, plant_p.alpha, plant_p.phi, RL_MAX_TILT_DEG)
    with torch.no_grad():
        mu_logstd = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        mu, _ = torch.chunk(mu_logstd, 2, dim=-1)
        action = torch.tanh(mu).squeeze(0).cpu().numpy().astype(np.float32)

    scales = np.r_[np.ones(RL_N_BASIS) * RL_SHAPE_ACTION_SCALE, RL_DURATION_ACTION_SCALE].astype(np.float32)
    params = np.clip(action, -1.0, 1.0) * scales
    base_ref, base_T = explicit_lqr_reference(theta0, theta_goal, plant_p, nom, lqr_w, duration=None)
    return decode_rl_reference(theta0, theta_goal, params, base_ref, base_T, plant_p.dt)


def build_trajectory_reference(trj_type: str, theta0: float, theta_goal: float, plant_p: PlantParams, nom: NominalModel, lqr_w: LQRWeights) -> Optional[Dict[str, object]]:
    kind = str(trj_type).upper()
    if kind == "LQR":
        if LQR_DURATION_S is None:
            return None
        return {"kind": "LQR", "duration": float(LQR_DURATION_S)}
    if kind == "NN":
        theta_ref, T = load_nn_reference(theta0, theta_goal, plant_p)
        return {"kind": "NN", "theta": theta_ref, "duration": T}
    if kind == "RL":
        theta_ref, T = load_rl_reference(theta0, theta_goal, plant_p, nom, lqr_w)
        return {"kind": "RL", "theta": theta_ref, "duration": T}
    raise ValueError("TRJ_TYPE must be 'LQR', 'NN', or 'RL'")


#%% ========================= PURE TDE+SMC ROLLOUT =========================

def rollout_once(
    plant: OneDOFRotorPlant,
    nom: NominalModel,
    task: Task,
    lqr_w: LQRWeights,
    smc_cfg: SMCConfig,
    cost_cfg: CostConfig,
    agent: Optional[object] = None,
    reference: Optional[Dict[str, object]] = None,
    seed: int = 0,
    collect_logs: bool = False,
):
    if agent is not None:
        raise ValueError("Residual torque agents were removed. Use agent=None and choose TRJ_TYPE instead.")

    np.random.seed(seed)
    dt = plant.p.dt
    horizon_s = time_horizon(task.theta0, task.theta_goal)
    ref_opts = {} if reference is None else dict(reference)
    if "duration" in ref_opts:
        horizon_s = float(ref_opts["duration"])
    N = max(1, int(round(horizon_s / dt)))

    ref_kind = str(ref_opts.get("kind", "LQR")).upper()
    has_theta = "theta" in ref_opts
    if (not has_theta) and ref_kind in ("LQR", "ILQR", "OPTIMIZED"):
        reference_kind = "LQR"
        x_ref, _ = generate_reference_ilqr_like(
            nom, plant.p,
            np.array([task.theta0, task.omega0], dtype=float),
            np.array([task.theta_goal, 0.0], dtype=float),
            N, dt, lqr_w,
        )
        theta_ref = np.concatenate([[task.theta0], x_ref[:-1, 0]])
    elif has_theta:
        reference_kind = ref_kind
        theta_seq = np.asarray(ref_opts["theta"], dtype=float)
        if len(theta_seq) == N + 1:
            theta_ref = theta_seq.copy()
        elif len(theta_seq) == N:
            theta_ref = np.empty(N + 1, dtype=float)
            theta_ref[0] = task.theta0
            theta_ref[1:] = theta_seq
        else:
            raise ValueError(f"Custom theta trajectory length {len(theta_seq)} does not match expected {N} or {N + 1}")
        theta_ref[0] = task.theta0
    else:
        raise ValueError(f"Unknown reference kind: {ref_kind}")

    omega_ref = finite_diff(theta_ref, dt)
    alpha_ref = finite_diff(omega_ref, dt)

    smc = TDE_SMC_Discrete(nom.J, nom.b, plant.p.dt, plant.p.u_max, plant.p.tau_i, plant.p.K_t, smc_cfg)
    smc.subtract_gravity_in_ueq = plant.p.subtract_gravity_in_ueq
    smc.reset()
    plant.reset(theta0=task.theta0, omega0=task.omega0)

    if collect_logs:
        t_log = np.zeros(N)
        th_ref_log = np.zeros(N)
        om_ref_log = np.zeros(N)
        al_ref_log = np.zeros(N)
        th_log = np.zeros(N)
        om_log = np.zeros(N)
        tau_m_log = np.zeros(N)
        u_rl_log = np.zeros(N)
        u_eq_log = np.zeros(N)
        u_s_log = np.zeros(N)
        s_log = np.zeros(N)
        eta_hat_log = np.zeros(N)
        d_hat_log = np.zeros(N)
        u_tde_log = np.zeros(N)
        u_smc_log = np.zeros(N)
        u_total_log = np.zeros(N)
        dist_log = np.zeros(N)
        gravity_log = np.zeros(N)

    total_cost = 0.0
    done = False
    steps_taken = 0
    t = 0.0

    for k in range(N):
        theta, omega, tau_m = plant.state.copy()
        theta_ref_k = theta_ref[k]
        theta_ref_k1 = theta_ref[min(k + 1, N)]
        omega_ref_k = omega_ref[k]
        omega_ref_k1 = omega_ref[min(k + 1, N)]

        u_cmd, info = smc.control(theta, omega, theta_ref_k, theta_ref_k1, omega_ref_k, omega_ref_k1, u_rl=0.0, plant=plant)
        plant.step(u_cmd)

        e = info["e"]
        edot = info["edot"]
        stage = cost_cfg.w_e * e * e + cost_cfg.w_edot * edot * edot + cost_cfg.w_omega * omega * omega
        total_cost += stage * dt

        theta2, omega2, _ = plant.state.copy()
        if abs(theta2 - task.theta_goal) < cost_cfg.goal_tol and abs(omega2) < cost_cfg.goal_tol:
            done = True

        if collect_logs:
            t_log[k] = t
            th_ref_log[k] = theta_ref_k
            om_ref_log[k] = omega_ref_k
            al_ref_log[k] = alpha_ref[k]
            th_log[k] = theta
            om_log[k] = omega
            tau_m_log[k] = tau_m
            u_rl_log[k] = 0.0
            u_eq_log[k] = info["u_eq"]
            u_s_log[k] = info["u_s"]
            s_log[k] = info["s"]
            eta_hat_log[k] = info["eta_hat"]
            d_hat_log[k] = info["d_hat"]
            u_tde_log[k] = -info["d_hat"]
            u_smc_log[k] = info["u_smc"]
            u_total_log[k] = info["u_total"]
            dist_log[k] = plant.last_disturbance
            gravity_log[k] = plant.last_gravity

        steps_taken = k + 1
        if done:
            break
        t += dt

    metrics = dict(total_cost=total_cost, finished=1.0 if done else 0.0, time=t)
    if not collect_logs:
        return metrics, None

    steps = steps_taken
    logs = dict(
        t=t_log[:steps], theta_ref=th_ref_log[:steps], omega_ref=om_ref_log[:steps], alpha_ref=al_ref_log[:steps],
        theta=th_log[:steps], omega=om_log[:steps], tau_m=tau_m_log[:steps], u_rl=u_rl_log[:steps],
        u_eq=u_eq_log[:steps], u_s=u_s_log[:steps], s=s_log[:steps], eta_hat=eta_hat_log[:steps],
        d_hat=d_hat_log[:steps], u_tde=u_tde_log[:steps], u_smc=u_smc_log[:steps], u_total=u_total_log[:steps],
        disturbance=dist_log[:steps], gravity=gravity_log[:steps], reference_kind=reference_kind,
    )
    return metrics, logs


def evaluate_and_rollout(trj_type: str = TRJ_TYPE) -> Dict[str, np.ndarray]:
    plant_p, nom, lqr_w, smc_cfg, cost_cfg = build_system_from_settings()
    theta0 = math.radians(THETA0_DEG)
    theta_goal = math.radians(THETA_GOAL_DEG)
    reference = build_trajectory_reference(trj_type, theta0, theta_goal, plant_p, nom, lqr_w)
    task = Task(theta0=theta0, omega0=0.0, theta_goal=theta_goal)
    plant = OneDOFRotorPlant(plant_p)
    metrics, logs = rollout_once(plant, nom, task, lqr_w, smc_cfg, cost_cfg, agent=None, reference=reference, seed=SEED, collect_logs=True)
    logs["metrics"] = metrics
    logs["trj_type"] = str(trj_type).upper()
    return logs


#%% ========================= PLOT HELPER =========================

def plot_rollout(logs: Dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt
    plant_p, _, _, _, _ = build_system_from_settings()
    t = logs["t"]
    theta = logs["theta"]
    theta_ref = logs["theta_ref"]
    omega = logs["omega"]
    omega_ref = logs["omega_ref"]
    u_total = logs["u_total"]
    tau_m = logs["tau_m"]
    s = logs["s"]
    d_hat = logs["d_hat"]
    theta0 = math.radians(THETA0_DEG)
    theta_goal = math.radians(THETA_GOAL_DEG)
    trj_type = logs.get("trj_type", logs.get("reference_kind", TRJ_TYPE))

    J_u = float(np.sum(u_total ** 2) * plant_p.dt)
    final_error = float(theta[-1] - theta_goal)
    print(f"Trajectory type = {trj_type}")
    print(f"Total rollout cost = {logs['metrics']['total_cost']:.6g}")
    print(f"Energy metric J_u = {J_u:.6g} N.m^2.s")
    print(f"Final angle error = {final_error:.6f} rad ({math.degrees(final_error):.3f} deg)")

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 10))
    fig.suptitle(f"TDE+SMC tracking with {trj_type} reference")
    axes[0].plot(t, theta, label="theta")
    axes[0].plot(t, theta_ref, "--", label="theta_ref")
    axes[0].axhline(theta0, linestyle=":", label="theta0")
    axes[0].axhline(theta_goal, linestyle="-.", label="theta_goal")
    axes[0].set_ylabel("theta [rad]")
    axes[0].legend(loc="best")

    axes[1].plot(t, omega, label="omega")
    axes[1].plot(t, omega_ref, "--", label="omega_ref")
    axes[1].set_ylabel("omega [rad/s]")
    axes[1].legend(loc="best")

    axes[2].plot(t, u_total, label="u command")
    axes[2].plot(t, tau_m, label="tau_m shaft")
    axes[2].set_ylabel("Torque / command")
    axes[2].legend(loc="best")

    axes[3].plot(t, s, label="s sliding")
    axes[3].plot(t, d_hat, label="d_hat")
    axes[3].set_ylabel("SMC/TDE")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend(loc="best")

    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


#%% ========================= RUN ROLLOUT =========================
# In Spyder, run USER SETTINGS first, then run this cell.
# The guard prevents this cell from running when trajectory_rl_gps_addon.py or
# true_gps_lqr_guided_addon.py imports this module for training.
logs_demo = evaluate_and_rollout(TRJ_TYPE)
print(f"Trajectory type = {TRJ_TYPE}")
print(f"Reference kind = {logs_demo.get('reference_kind')}")
print(f"Total cost = {logs_demo['metrics']['total_cost']:.6g}")


#%% ========================= PLOT ROLLOUT =========================
# Run this cell after RUN ROLLOUT.
if __name__ == "__main__":
    plot_rollout(logs_demo)
