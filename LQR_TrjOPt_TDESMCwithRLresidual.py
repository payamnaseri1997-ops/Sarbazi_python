# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 01:12:30 2025

@author: elecomp
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import os, math
import numpy as np

# -------------------------
# Utilities
# -------------------------

def sat(x: float, limit: float) -> float:
    """Saturate scalar x into [-limit, +limit]."""
    return max(-limit, min(limit, x))

def sat_vec(x: np.ndarray, limit: float) -> np.ndarray:
    """Elementwise saturation of vector x into [-limit, +limit]."""
    return np.clip(x, -limit, limit)

def finite_diff(y: np.ndarray, dt: float) -> np.ndarray:
    """Compute derivative using forward/backward at ends and centered inside."""
    dy = np.zeros_like(y)
    if len(y) >= 2:
        dy[0] = (y[1] - y[0]) / dt
        dy[-1] = (y[-1] - y[-2]) / dt
    if len(y) >= 3:
        dy[1:-1] = (y[2:] - y[:-2]) / (2.0 * dt)
    return dy

# -------------------------
# Plant and nominal model definitions
# -------------------------

@dataclass
class PlantParams:
    """True plant + actuator parameters used inside the simulator.

    Parameters
    ----------
    J : float
        Rotor inertia [kg·m²].
    b : float
        Viscous damping [N·m·s/rad].
    u_max : float
        Servo command magnitude limit (torque reference) [N·m].
    omega_max : float
        Soft speed limit used in cost function [rad/s].
    dt : float
        Control / integration time step [s].
    tau_i : float
        Time constant of the first-order current/torque loop [s].
    K_t : float
        Motor torque constant translating command units to [N·m].
    m1 : float, optional
        Grey base mass contributing to inertia [kg].
    R1 : float, optional
        Grey base radius measured from the spin axis [m].
    m2 : float, optional
        Blue post mass located at offset ``r1`` [kg].
    a2 : float, optional
        Blue post local radius used for its own polar inertia [m].
    r1 : float, optional
        Blue post radial offset from the spin axis [m].
    m3 : float, optional
        Orange post mass located at offset ``r2`` [kg].
    a3 : float, optional
        Orange post local radius [m].
    r2 : float, optional
        Orange post radial offset from the spin axis [m].
    m4 : float, optional
        Red link mass treated as a slender bar [kg].
    L4 : float, optional
        Red link total length for the in-plane inertia calculation [m].
    l4_c : float, optional
        Red link centre-of-mass distance from the joint along the link [m].
    gamma : float, optional
        Red link COM azimuth in the plane relative to the orange post [rad].
    beta2 : float, optional
        In-plane azimuth of the blue post COM relative to the spin axis [rad].
    beta4 : float, optional
        In-plane azimuth of the red link COM relative to the spin axis [rad].
    alpha : float, optional
        Platform roll tilt angle (rotation about x) [rad].
    phi : float, optional
        Platform pitch tilt angle (rotation about y) [rad].
    g : float, optional
        Gravity magnitude used for tilt loading [m/s²].
    subtract_gravity_in_ueq : bool, optional
        Enable pre-cancellation of gravity inside ``u_eq`` when ``True``.
    """

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
#%%
@dataclass
class NominalModel:
    """Nominal model parameters used in controller/trajectory computations."""
    J: float
    b: float


def J_local_cylinder_z(m: float, a: float) -> float:
    """Polar moment about the local cylinder axis (solid)."""
    return 0.5 * m * a * a


def J_bar_inplane_COM(m: float, L: float) -> float:
    """Polar inertia of a slender bar in-plane about its COM."""
    return (1.0 / 12.0) * m * L * L


def red_link_radius_from_pin(r2: float, l_c: float, gamma: float) -> float:
    """Horizontal distance from base axis to the red link COM."""
    return math.sqrt(max(0.0, r2 * r2 + l_c * l_c + 2.0 * r2 * l_c * math.cos(gamma)))


def compute_equivalent_inertia(p: PlantParams) -> float:
    """Combine geometry contributions into the rotor equivalent inertia."""
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
    """Project gravity into the rotor plane given base tilt angles."""
    gx = -g * math.sin(phi)
    gy = g * math.sin(alpha) * math.cos(phi)
    psi = math.atan2(gy, gx)
    g_t = math.hypot(gx, gy)
    return g_t, psi


class OneDOFRotorPlant:
    """True 1-DoF rotational system with a first-order torque servo.

    State vector: ``x = [theta, omega, tau_m]`` where ``tau_m`` is the actual
    shaft torque generated by the actuator.  The control input is the servo
    torque command ``u`` that is filtered by the first-order loop.

    Mechanical dynamics:

    ``omega[k+1] = omega[k] + dt/J * (tau_m[k] - b * omega[k] + d[k])``
    ``theta[k+1] = theta[k] + dt * omega[k+1]``

    Servo current/torque loop:

    ``tau_m[k+1] = tau_m[k] + dt * (-(1/tau_i) * tau_m[k] + (K_t/tau_i) * u[k])``

    Disturbance ``d[k]`` follows the same structure as the previous
    implementation (Coulomb friction, periodic loads, slowly varying
    components, and white torque noise).
    """

    def __init__(self, p: PlantParams):
        self.p = p
        self.state = np.zeros(3)  # [theta, omega, tau_m]
        self.J_eq = compute_equivalent_inertia(self.p)
        # Disturbance parameters (tune as needed)
        self.torque_coulomb = 0.02  # Nm
        self.load_amp = 0.03        # Nm
        self.load_freq = 1.5        # Hz
        self.noise_std = 0.01       # Nm
        # Additional unmodeled disturbance with varying amplitude/frequency
        self.extra_base_amp = 0.02  # Nm
        self.extra_base_freq = 2.0  # Hz
        self.extra_amp_mod = 0.5    # relative amplitude variation
        self.extra_amp_mod_freq = 0.1  # Hz, amplitude modulation frequency
        self.extra_freq_mod = 0.5   # relative frequency variation
        self.extra_freq_mod_freq = 0.05  # Hz, frequency modulation frequency
        self.time = 0.0
        self.last_disturbance = 0.0
        self.last_tau_m = 0.0
        self.last_command = 0.0
        self.last_gravity = 0.0

    def gravity_torque(self, theta: float) -> float:
        g_t, psi = gravity_proj_inplane(self.p.alpha, self.p.phi, self.p.g)
        R4 = (
            red_link_radius_from_pin(self.p.r2, self.p.l4_c, self.p.gamma)
            if self.p.m4 > 0
            else 0.0
        )
        tau2 = (
            self.p.m2 * self.p.r1 * g_t * math.sin(theta + self.p.beta2 - psi)
            if self.p.m2 > 0
            else 0.0
        )
        tau4 = (
            self.p.m4 * R4 * g_t * math.sin(theta + self.p.beta4 - psi)
            if self.p.m4 > 0
            else 0.0
        )
        return tau2 + tau4

    def reset(
        self,
        theta0: float = 0.0,
        omega0: float = 0.0,
        tau_m0: float = 0.0,
    ) -> np.ndarray:
        """Reset the plant state.
        theta0: initial angle [rad]
        omega0: initial angular velocity [rad/s]
        tau_m0: initial actuator torque [N·m]
        Returns the current state copy.
        """
        self.state[:] = [theta0, omega0, tau_m0]
        self.J_eq = compute_equivalent_inertia(self.p)
        self.time = 0.0
        self.last_disturbance = 0.0
        self.last_tau_m = tau_m0
        self.last_command = 0.0
        self.last_gravity = 0.0
        return self.state.copy()

    def step(self, u_cmd: float) -> np.ndarray:
        """Advance dynamics one time step with applied torque u_cmd.
        Applies saturation to respect u_max and integrates with semi-implicit Euler.
        Returns the updated state copy.
        """
        u = sat(u_cmd, self.p.u_max)
        theta, omega, tau_m = self.state
        # Disturbances
        d_coul = self.torque_coulomb * (1.0 if omega >= 0 else -1.0) if abs(omega) > 1e-5 else 0.0
        d_per = self.load_amp * math.sin(2.0 * math.pi * self.load_freq * self.time)
        amp_var = self.extra_base_amp * (
            1.0
            + self.extra_amp_mod
            * math.sin(2.0 * math.pi * self.extra_amp_mod_freq * self.time)
        )
        freq_var = self.extra_base_freq * (
            1.0
            + self.extra_freq_mod
            * math.sin(2.0 * math.pi * self.extra_freq_mod_freq * self.time)
        )
        d_var = amp_var * math.sin(2.0 * math.pi * freq_var * self.time)
        d_noise = np.random.randn() * self.noise_std
        tau_g = self.gravity_torque(theta)
        d = d_coul + d_per + d_var + d_noise + tau_g
        self.last_disturbance = d
        self.last_tau_m = tau_m
        self.last_command = u
        self.last_gravity = tau_g

        # Dynamics integration (semi-implicit Euler)
        domega = (tau_m - self.p.b * omega + d) / self.J_eq
        omega_next = omega + domega * self.p.dt
        theta_next = theta + omega_next * self.p.dt
        tau_m_next = tau_m + self.p.dt * (
            -(1.0 / self.p.tau_i) * tau_m + (self.p.K_t / self.p.tau_i) * u
        )
        self.state[:] = [theta_next, omega_next, tau_m_next]
        self.time += self.p.dt
        return self.state.copy()

# -------------------------
# iLQR-like finite-horizon LQR to generate reference trajectory
# -------------------------

@dataclass
class LQRWeights:
    """Weights for the LQR/iLQR reference generation on the nominal model."""
    q_theta: float
    q_omega: float
    r_u: float
    qT_theta: float
    qT_omega: float
    omega_limit_penalty: float

def build_AB(nom: NominalModel, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Discrete-time linear model x_{k+1} = A x_k + B u_k for x=[theta, omega]."""
    J, b = nom.J, nom.b
    A = np.array([[1.0, dt],
                  [0.0, 1.0 - (b/J)*dt]])
    B = np.array([[0.0],
                  [dt / J]])
    return A, B

def finite_horizon_lqr_gain(A: np.ndarray, B: np.ndarray, N: int, w: LQRWeights) -> List[np.ndarray]:
    """Solve the backward Riccati recursion to get time-varying gains K[k]."""
    Q = np.diag([w.q_theta, w.q_omega])
    R = np.array([[w.r_u]])
    Qf = np.diag([w.qT_theta, w.qT_omega])

    P = Qf.copy()
    Ks: List[np.ndarray] = [np.zeros((1, 2)) for _ in range(N)]
    for k in reversed(range(N)):
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)  # 1x2
        Ks[k] = K
        P = Q + A.T @ P @ (A - B @ K)
    return Ks

def generate_reference_ilqr_like(
    nom: NominalModel,
    plant_limits: PlantParams,
    x0: np.ndarray,
    x_goal: np.ndarray,
    N: int,
    dt: float,
    w: LQRWeights
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a feasible reference (x_ref, u_ref) using finite-horizon LQR on nominal model.
    - Torque is clamped to u_max (hard input constraint).
    - Speed beyond omega_max is penalized with omega_limit_penalty.
    Returns:
      x_ref: (N+1, 2) states [theta, omega]
      u_ref: (N,) torque sequence
    """
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

# -------------------------
# TDE + Sliding Mode Controller
# -------------------------

@dataclass
class SMCConfig:
    """Sliding Mode Controller (SMC) + TDE tuning parameters.
    lambda_s: surface slope (>0). Higher = faster convergence but more control effort.
    k: sliding gain (>0). Higher = stronger attraction to surface; too high may chatter.
    phi: boundary layer half-width for smooth sat (tanh). Larger = smoother, more steady-state error.
    """
    lambda_s: float
    k: float
    phi: float

class TDE_SMC_Discrete:
    """Discrete-time TDE + SMC controller aware of actuator dynamics."""

    def __init__(
        self,
        hatJ: float,
        hatb: float,
        dt: float,
        u_max: float,
        tau_i: float,
        K_t: float,
        smc: SMCConfig,
    ):
        self.hatJ = hatJ
        self.hatb = hatb
        self.dt = dt
        self.u_max = u_max
        self.tau_i = tau_i
        self.K_t = K_t
        self.cfg = smc
        self.tau_m_hat: float = 0.0
        self.omega_hist: List[float] = []
        self.tau_m_hat_hist: List[float] = []
        self.subtract_gravity_in_ueq: bool = False

    def reset(self):
        self.tau_m_hat = 0.0
        self.omega_hist.clear()
        self.tau_m_hat_hist.clear()

    def control(
        self,
        theta: float,
        omega: float,
        theta_ref_k: float,
        theta_ref_k1: float,
        omega_ref_k: float,
        omega_ref_k1: float,
        u_rl: float = 0.0,
        plant: Optional[OneDOFRotorPlant] = None,
    ) -> Tuple[float, Dict[str, float]]:
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
        u_eq = self.hatb * omega + (self.hatJ / denom) * (
            domega_ref - self.cfg.lambda_s * (self.dt * omega - dtheta_ref)
        )
        if getattr(self, "subtract_gravity_in_ueq", False) and plant is not None:
            u_eq = u_eq - plant.gravity_torque(theta)

        s_norm = s / (self.cfg.phi + 1e-9)
        u_s = -self.cfg.k * np.tanh(s_norm)

        u_total = sat(u_eq - d_hat + u_s + u_rl, self.u_max)

        tau_m_hat_prev = self.tau_m_hat
        tau_m_hat_next = tau_m_hat_prev + self.dt * (
            -(1.0 / self.tau_i) * tau_m_hat_prev + (self.K_t / self.tau_i) * u_total
        )
        self.tau_m_hat = tau_m_hat_next
        self.tau_m_hat_hist.append(tau_m_hat_next)
        if len(self.tau_m_hat_hist) > 2:
            self.tau_m_hat_hist.pop(0)

        info = dict(
            e=e,
            edot=edot,
            s=s,
            eta_hat=eta_hat,
            d_hat=d_hat,
            u_eq=u_eq,
            u_s=u_s,
            u_smc=u_eq - d_hat + u_s,
            u_total=u_total,
        )
        return u_total, info

# -------------------------
# Rollout cost configuration
# -------------------------

@dataclass
class CostConfig:
    """Weights for the per-step rollout cost."""
    w_e: float
    w_edot: float
    w_u: float
    w_omega: float
    goal_tol: float
    done_bonus: float

@dataclass
class Task:
    """Task specification: start and goal for one episode."""
    theta0: float
    omega0: float
    theta_goal: float


def time_horizon(theta0: float, theta_goal: float) -> float:
    """Compute time horizon based on angular difference.

    The horizon scales linearly with |θ_goal−θ_0| such that a π radian
    move takes 4 seconds and a π/2 radian move takes 2 seconds.
    """
    return 4.0 * abs(theta_goal - theta0) / math.pi


# -------------------------
# Configuration containers only
# -------------------------
# These dataclasses intentionally do NOT contain numeric defaults.
# The only user-editable numeric values are in the USER SETTINGS cell below.

@dataclass
class PlantConfig:
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
    alpha_deg: float
    phi_deg: float
    g: float
    subtract_gravity_in_ueq: bool

@dataclass
class NominalConfig:
    J: float
    b: float

@dataclass
class LQRConfig:
    q_theta: float
    q_omega: float
    r_u: float
    qT_theta: float
    qT_omega: float
    omega_limit_penalty: float

@dataclass
class ControllerConfig:
    lambda_s: float
    k: float
    phi: float

@dataclass
class RolloutCostConfig:
    w_e: float
    w_edot: float
    w_u: float
    w_omega: float
    goal_tol: float
    done_bonus: float

@dataclass
class NNTrajectoryConfig:
    weights_dir: str
    weights_file: str
    n_shape: int
    hidden_sizes: Tuple[int, int]
    omega_ref_limit: float
    min_duration: float
    max_duration: float
    max_tilt_deg: float

@dataclass
class RLTrajectoryConfig:
    weights_dir: str
    checkpoint_file: str
    actor_file: str
    n_basis: int
    hidden_sizes: Tuple[int, int]
    shape_action_scale: float
    duration_action_scale: float
    omega_ref_limit: float
    duration_margin: float
    max_duration_factor: float
    min_extra_duration: float
    duration_exp_scale: float
    z_grid_size: int
    max_tilt_deg: float

@dataclass
class ExperimentConfig:
    trj_type: str
    theta0_deg: float
    theta_goal_deg: float
    seed: int
    lqr_duration_s: Optional[float]
    plant: PlantConfig
    nominal: NominalConfig
    lqr: LQRConfig
    controller: ControllerConfig
    cost: RolloutCostConfig
    nn: NNTrajectoryConfig
    rl: RLTrajectoryConfig

#%% ========================= USER SETTINGS =========================
# Edit ONLY this cell for normal experiments.
# trj_type chooses the reference trajectory generator: "LQR", "NN", or "RL".
CFG = ExperimentConfig(
    trj_type="LQR",
    theta0_deg=0.0,
    theta_goal_deg=-90.0,
    seed=0,
    # LQR duration override. This restores the old behavior where demo_duration=150.2
    # was passed as reference={"duration": demo_duration}. Set to None for automatic
    # time_horizon(theta0, theta_goal).
    lqr_duration_s=150.2,

    plant=PlantConfig(
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
        alpha_deg=25.0,
        phi_deg=3.0,
        g=9.81,
        subtract_gravity_in_ueq=False,
    ),

    nominal=NominalConfig(
        J=13000.0,
        b=0.06,
    ),

    lqr=LQRConfig(
        q_theta=85.0,
        q_omega=18.0,
        r_u=0.02,
        qT_theta=4200.0,
        qT_omega=220.0,
        omega_limit_penalty=1000.0,
    ),

    controller=ControllerConfig(
        lambda_s=85.0,
        k=10.85,
        phi=0.025,
    ),

    cost=RolloutCostConfig(
        w_e=8.0,
        w_edot=1.0,
        w_u=0.03,
        w_omega=0.3,
        goal_tol=1e-2,
        done_bonus=2.0,
    ),

    nn=NNTrajectoryConfig(
        weights_dir="trajectory_rl_results",
        weights_file="trajectory_policy.pt",
        n_shape=5,
        hidden_sizes=(64, 64),
        omega_ref_limit=0.2,
        min_duration=0.5,
        max_duration=80.0,
        max_tilt_deg=20.0,
    ),

    rl=RLTrajectoryConfig(
        weights_dir="true_gps_results_smoke",
        checkpoint_file="sac_gps_agent.pt",
        actor_file="sac_gps_actor.pt",
        n_basis=6,
        hidden_sizes=(128, 128),
        shape_action_scale=1.30,
        duration_action_scale=1.20,
        omega_ref_limit=0.2,
        duration_margin=1.05,
        max_duration_factor=2.5,
        min_extra_duration=5.0,
        duration_exp_scale=0.35,
        z_grid_size=1201,
        max_tilt_deg=20.0,
    ),
)
#%%
def build_system_from_config(cfg: ExperimentConfig):
    pc = cfg.plant
    plant_p = PlantParams(
        J=pc.J, b=pc.b, u_max=pc.u_max, omega_max=pc.omega_max, dt=pc.dt,
        tau_i=pc.tau_i, K_t=pc.K_t, m1=pc.m1, R1=pc.R1, m2=pc.m2,
        a2=pc.a2, r1=pc.r1, m3=pc.m3, a3=pc.a3, r2=pc.r2,
        m4=pc.m4, L4=pc.L4, l4_c=pc.l4_c, gamma=pc.gamma,
        beta2=pc.beta2, beta4=pc.beta4, alpha=math.radians(pc.alpha_deg),
        phi=math.radians(pc.phi_deg), g=pc.g,
        subtract_gravity_in_ueq=pc.subtract_gravity_in_ueq,
    )
    nom = NominalModel(J=cfg.nominal.J, b=cfg.nominal.b)
    lqr_w = LQRWeights(
        q_theta=cfg.lqr.q_theta, q_omega=cfg.lqr.q_omega, r_u=cfg.lqr.r_u,
        qT_theta=cfg.lqr.qT_theta, qT_omega=cfg.lqr.qT_omega,
        omega_limit_penalty=cfg.lqr.omega_limit_penalty,
    )
    smc_cfg = SMCConfig(lambda_s=cfg.controller.lambda_s, k=cfg.controller.k, phi=cfg.controller.phi)
    cost_cfg = CostConfig(
        w_e=cfg.cost.w_e, w_edot=cfg.cost.w_edot, w_u=cfg.cost.w_u,
        w_omega=cfg.cost.w_omega, goal_tol=cfg.cost.goal_tol,
        done_bonus=cfg.cost.done_bonus,
    )
    return plant_p, nom, lqr_w, smc_cfg, cost_cfg


def _torch_import():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError(f"PyTorch is required for trj_type='NN' or 'RL': {exc}")
    return torch, nn


def _trapz(y, x):
    y = np.asarray(y, dtype=float); x = np.asarray(x, dtype=float)
    return 0.0 if len(y) < 2 else float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def _cumtrapz(y, x):
    y = np.asarray(y, dtype=float); x = np.asarray(x, dtype=float)
    out = np.zeros_like(y, dtype=float)
    if len(y) >= 2:
        out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * np.diff(x))
    return out


def _sin_basis(z, n):
    return np.stack([np.sin(k * math.pi * z) for k in range(1, n + 1)], axis=1)


def _softplus(x):
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _minimum_jerk(z):
    return 10.0 * z**3 - 15.0 * z**4 + 6.0 * z**5


def explicit_lqr_reference(theta0, theta_goal, plant_p, nom, lqr_w):
    T = time_horizon(theta0, theta_goal)
    N = max(1, int(round(T / plant_p.dt)))
    T = N * plant_p.dt
    x_ref, _ = generate_reference_ilqr_like(
        nom, plant_p,
        np.array([theta0, 0.0], dtype=float),
        np.array([theta_goal, 0.0], dtype=float),
        N, plant_p.dt, lqr_w,
    )
    return np.concatenate([[theta0], x_ref[:-1, 0]]), T


def _case_features(theta_goal, alpha, phi, max_tilt_deg):
    max_tilt = math.radians(max_tilt_deg)
    return np.array([theta_goal / math.pi, alpha / max_tilt, phi / max_tilt], dtype=np.float32)


def _decode_nn_reference(theta0, theta_goal, params, dt, cfg: NNTrajectoryConfig):
    params = np.asarray(params, dtype=float).reshape(-1)
    delta = theta_goal - theta0
    if abs(delta) < 1e-12:
        t = np.arange(0.0, cfg.min_duration + 0.5 * dt, dt)
        return np.full_like(t, theta0), float(t[-1])
    z = np.linspace(0.0, 1.0, 1001)
    logits = _sin_basis(z, cfg.n_shape) @ params[:cfg.n_shape]
    v = z * (1.0 - z) * np.exp(np.clip(logits, -5.0, 5.0)) + 1e-6
    v /= max(_trapz(v, z), 1e-12)
    h = _cumtrapz(v, z); h /= max(h[-1], 1e-12)
    T_min = abs(delta) * float(np.max(v)) / max(cfg.omega_ref_limit, 1e-12)
    T = float(np.clip(T_min + float(_softplus([params[-1]])[0]), cfg.min_duration, cfg.max_duration))
    N = max(1, int(round(T / dt)))
    t = np.arange(N + 1) * dt; T = float(t[-1])
    theta_ref = theta0 + delta * np.interp(t / max(T, 1e-12), z, h)
    theta_ref[0] = theta0; theta_ref[-1] = theta_goal
    return theta_ref, T


def load_nn_reference(theta0, theta_goal, plant_p, cfg: NNTrajectoryConfig):
    torch, nn = _torch_import()
    path = os.path.join(cfg.weights_dir, cfg.weights_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"NN weights not found: {path}")
    h1, h2 = cfg.hidden_sizes
    net = nn.Sequential(nn.Linear(3, h1), nn.Tanh(), nn.Linear(h1, h2), nn.Tanh(), nn.Linear(h2, cfg.n_shape + 1))
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    x = torch.as_tensor(_case_features(theta_goal, plant_p.alpha, plant_p.phi, cfg.max_tilt_deg), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        params = net(x).squeeze(0).cpu().numpy().astype(float)
    params[:-1] = np.clip(params[:-1], -4.0, 4.0); params[-1] = np.clip(params[-1], -4.0, 12.0)
    return _decode_nn_reference(theta0, theta_goal, params, plant_p.dt, cfg)


def _lqr_guide_shape(theta_ref, theta_goal, n_grid):
    z_old = np.linspace(0.0, 1.0, len(theta_ref)); z = np.linspace(0.0, 1.0, n_grid)
    raw = np.asarray(theta_ref, dtype=float) - float(theta_ref[0])
    scale = float(raw[-1]) if len(raw) else 0.0
    if abs(theta_goal) < 1e-12:
        h = np.zeros_like(z)
    elif abs(scale) < 1e-9:
        h = _minimum_jerk(z)
    else:
        h_old = np.maximum.accumulate(np.clip(raw / scale, 0.0, 1.0))
        h_old[0] = 0.0; h_old[-1] = 1.0
        h = np.interp(z, z_old, h_old)
        h = np.maximum.accumulate(np.clip(0.85 * h + 0.15 * _minimum_jerk(z), 0.0, 1.0))
        h[0] = 0.0; h[-1] = 1.0
    v = np.maximum(np.gradient(h, z), 1e-6)
    v = np.minimum(v, max(5.0, 3.0 * float(np.percentile(v, 95))))
    v /= max(_trapz(v, z), 1e-12)
    return z, h, v


def _decode_rl_reference(theta0, theta_goal, params, base_ref, base_T, dt, cfg: RLTrajectoryConfig):
    params = np.asarray(params, dtype=float).reshape(-1)
    delta = theta_goal - theta0
    if abs(delta) < 1e-12:
        N = max(1, int(math.ceil(max(0.5, base_T) / dt)))
        return np.full(N + 1, theta0), N * dt
    z, _, v_base = _lqr_guide_shape(base_ref, theta_goal, cfg.z_grid_size)
    v = np.maximum(v_base, 1e-8) * np.exp(np.clip(_sin_basis(z, cfg.n_basis) @ params[:cfg.n_basis], -3.0, 3.0))
    v = np.maximum(v, 1e-9); v /= max(_trapz(v, z), 1e-12)
    T_min = cfg.duration_margin * abs(delta) / cfg.omega_ref_limit
    T_center = max(base_T, T_min)
    T_max = max(cfg.max_duration_factor * T_min, T_min + cfg.min_extra_duration, T_center)
    T = float(np.clip(T_center * math.exp(cfg.duration_exp_scale * float(params[-1])), T_min, T_max))
    T = max(T, cfg.duration_margin * abs(delta) * float(np.max(v)) / cfg.omega_ref_limit)
    N = max(1, int(math.ceil(T / dt))); T = N * dt
    h = _cumtrapz(v, z); h /= max(h[-1], 1e-12)
    h = np.maximum.accumulate(np.clip(h, 0.0, 1.0)); h[0] = 0.0; h[-1] = 1.0
    t = np.arange(N + 1) * dt
    theta_ref = theta0 + delta * np.interp(t / max(T, 1e-12), z, h)
    theta_ref[0] = theta0; theta_ref[-1] = theta_goal
    return theta_ref, T


def load_rl_reference(theta0, theta_goal, plant_p, nom, lqr_w, cfg: RLTrajectoryConfig):
    torch, nn = _torch_import()
    p1 = os.path.join(cfg.weights_dir, cfg.checkpoint_file)
    p2 = os.path.join(cfg.weights_dir, cfg.actor_file)
    path = p1 if os.path.exists(p1) else p2
    if not os.path.exists(path):
        raise FileNotFoundError(f"RL actor weights not found in {cfg.weights_dir}")
    act_dim = cfg.n_basis + 1; h1, h2 = cfg.hidden_sizes
    net = nn.Sequential(nn.Linear(3, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 2 * act_dim))
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("actor", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("net.net.") for k in state.keys()):
        state = {k.replace("net.net.", "", 1): v for k, v in state.items() if k.startswith("net.net.")}
    net.load_state_dict(state); net.eval()
    obs = _case_features(theta_goal, plant_p.alpha, plant_p.phi, cfg.max_tilt_deg)
    with torch.no_grad():
        mu_logstd = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        mu, _ = torch.chunk(mu_logstd, 2, dim=-1)
        action = torch.tanh(mu).squeeze(0).cpu().numpy().astype(np.float32)
    scales = np.r_[np.ones(cfg.n_basis) * cfg.shape_action_scale, cfg.duration_action_scale].astype(np.float32)
    params = np.clip(action, -1.0, 1.0) * scales
    base_ref, base_T = explicit_lqr_reference(theta0, theta_goal, plant_p, nom, lqr_w)
    return _decode_rl_reference(theta0, theta_goal, params, base_ref, base_T, plant_p.dt, cfg)


def build_trajectory_reference(trj_type, theta0, theta_goal, plant_p, nom, lqr_w, cfg: ExperimentConfig):
    kind = str(trj_type).upper()
    if kind == "LQR":
        # Keep LQR generation inside rollout_once, but optionally force the
        # same long horizon that the old demo used through reference={"duration": ...}.
        if cfg.lqr_duration_s is None:
            return None
        return {"kind": "LQR", "duration": float(cfg.lqr_duration_s)}
    if kind == "NN":
        theta_ref, T = load_nn_reference(theta0, theta_goal, plant_p, cfg.nn)
        return {"kind": "NN", "theta": theta_ref, "duration": T}
    if kind == "RL":
        theta_ref, T = load_rl_reference(theta0, theta_goal, plant_p, nom, lqr_w, cfg.rl)
        return {"kind": "RL", "theta": theta_ref, "duration": T}
    raise ValueError("trj_type must be 'LQR', 'NN', or 'RL'")


# -------------------------
# Pure TDE+SMC rollout: no residual agent code
# -------------------------

def rollout_once(
    plant: OneDOFRotorPlant,
    nom: NominalModel,
    task: Task,
    lqr_w: LQRWeights,
    smc_cfg: SMCConfig,
    cost_cfg: CostConfig,
    reference: Optional[Dict[str, object]] = None,
    seed: int = 0,
    collect_logs: bool = False,
):
    np.random.seed(seed)
    dt = plant.p.dt
    horizon_s = time_horizon(task.theta0, task.theta_goal)
    N = int(round(horizon_s / dt))
    ref_opts = {} if reference is None else dict(reference)
    if "duration" in ref_opts:
        horizon_s = float(ref_opts["duration"]); N = int(round(horizon_s / dt))
    ref_kind = str(ref_opts.get("kind", "LQR")).upper()
    has_theta = "theta" in ref_opts
    if (not has_theta) and ref_kind.lower() in ("lqr", "ilqr", "optimized"):
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
            theta_ref = np.empty(N + 1); theta_ref[0] = task.theta0; theta_ref[1:] = theta_seq
        else:
            raise ValueError(f"Custom theta trajectory length {len(theta_seq)} does not match expected {N} or {N + 1}")
        theta_ref[0] = task.theta0
    else:
        raise ValueError(f"Unknown reference kind: {ref_kind}")

    theta_ref = np.asarray(theta_ref, dtype=float)
    omega_ref = finite_diff(theta_ref, dt)
    alpha_ref = finite_diff(omega_ref, dt)

    smc = TDE_SMC_Discrete(nom.J, nom.b, plant.p.dt, plant.p.u_max, plant.p.tau_i, plant.p.K_t, smc_cfg)
    smc.subtract_gravity_in_ueq = plant.p.subtract_gravity_in_ueq
    smc.reset(); plant.reset(task.theta0, task.omega0)

    if collect_logs:
        logs = {k: np.zeros(N) for k in (
            "t", "theta_ref", "omega_ref", "alpha_ref", "theta", "omega", "tau_m",
            "u_rl", "u_eq", "u_s", "s", "eta_hat", "d_hat", "u_tde", "u_smc",
            "u_total", "disturbance", "gravity"
        )}
    total_cost = 0.0; done = False; steps = 0; t = 0.0
    for k in range(N):
        theta, omega, tau_m = plant.state.copy()
        u_cmd, info = smc.control(
            theta, omega,
            theta_ref[k], theta_ref[min(k + 1, N)],
            omega_ref[k], omega_ref[min(k + 1, N)],
            u_rl=0.0, plant=plant,
        )
        plant.step(u_cmd)
        e = info["e"]; edot = info["edot"]
        stage = cost_cfg.w_e * e * e + cost_cfg.w_edot * edot * edot + cost_cfg.w_omega * omega * omega
        total_cost += stage * dt
        theta2, omega2, _ = plant.state.copy()
        done = abs(theta2 - task.theta_goal) < cost_cfg.goal_tol and abs(omega2) < cost_cfg.goal_tol
        if collect_logs:
            logs["t"][k] = t; logs["theta_ref"][k] = theta_ref[k]; logs["omega_ref"][k] = omega_ref[k]
            logs["alpha_ref"][k] = alpha_ref[k]; logs["theta"][k] = theta; logs["omega"][k] = omega
            logs["tau_m"][k] = tau_m; logs["u_eq"][k] = info["u_eq"]; logs["u_s"][k] = info["u_s"]
            logs["s"][k] = info["s"]; logs["eta_hat"][k] = info["eta_hat"]; logs["d_hat"][k] = info["d_hat"]
            logs["u_tde"][k] = -info["d_hat"]; logs["u_smc"][k] = info["u_smc"]; logs["u_total"][k] = info["u_total"]
            logs["disturbance"][k] = plant.last_disturbance; logs["gravity"][k] = plant.last_gravity
        steps = k + 1
        if done: break
        t += dt
    metrics = {"total_cost": total_cost, "finished": 1.0 if done else 0.0, "time": t}
    if not collect_logs:
        return metrics, None
    logs = {k: v[:steps] for k, v in logs.items()}
    logs["reference_kind"] = reference_kind
    return metrics, logs


def evaluate_and_rollout(cfg: ExperimentConfig = CFG, reference: Optional[Dict[str, object]] = None):
    plant_p, nom, lqr_w, smc_cfg, cost_cfg = build_system_from_config(cfg)
    theta0 = math.radians(cfg.theta0_deg); theta_goal = math.radians(cfg.theta_goal_deg)
    if reference is None:
        reference = build_trajectory_reference(cfg.trj_type, theta0, theta_goal, plant_p, nom, lqr_w, cfg)
    metrics, logs = rollout_once(
        plant=OneDOFRotorPlant(plant_p), nom=nom, task=Task(theta0, 0.0, theta_goal),
        lqr_w=lqr_w, smc_cfg=smc_cfg, cost_cfg=cost_cfg,
        reference=reference, seed=cfg.seed, collect_logs=True,
    )
    logs["metrics"] = metrics; logs["trj_type"] = str(cfg.trj_type).upper()
    logs["theta0"] = theta0; logs["theta_goal"] = theta_goal
    return logs


def plot_rollout(logs):
    import matplotlib.pyplot as plt
    t = logs["t"]; trj_type = logs.get("trj_type", logs.get("reference_kind", "LQR"))
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 10))
    fig.suptitle(f"TDE+SMC tracking with {trj_type} reference")
    axes[0].plot(t, logs["theta"], label="theta"); axes[0].plot(t, logs["theta_ref"], "--", label="theta_ref")
    axes[0].axhline(logs["theta0"], linestyle=":", label="theta0"); axes[0].axhline(logs["theta_goal"], linestyle="-.", label="theta_goal")
    axes[0].set_ylabel("theta [rad]"); axes[0].legend(loc="best")
    axes[1].plot(t, logs["omega"], label="omega"); axes[1].plot(t, logs["omega_ref"], "--", label="omega_ref")
    axes[1].set_ylabel("omega [rad/s]"); axes[1].legend(loc="best")
    axes[2].plot(t, logs["u_total"], label="u command"); axes[2].plot(t, logs["tau_m"], label="tau_m shaft")
    axes[2].set_ylabel("torque / command"); axes[2].legend(loc="best")
    axes[3].plot(t, logs["s"], label="s sliding"); axes[3].plot(t, logs["d_hat"], label="d_hat")
    axes[3].set_ylabel("SMC/TDE"); axes[3].set_xlabel("time [s]"); axes[3].legend(loc="best")
    for ax in axes: ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()



#%% ========================= RUN ROLLOUT =========================
# In Spyder: run the USER SETTINGS cell first if you changed anything,
# then run this cell. No main() function is used.
logs_demo = evaluate_and_rollout(CFG)
plant_p, *_ = build_system_from_config(CFG)
J_u = float(np.sum(logs_demo["u_total"] ** 2) * plant_p.dt)
final_error = float(logs_demo["theta"][-1] - logs_demo["theta_goal"])
print(f"Trajectory type = {CFG.trj_type}")
print(f"Energy metric J_u = {J_u:.4f}")
print(f"Final angle error = {final_error:.6f} rad ({math.degrees(final_error):.3f} deg)")

#%% ========================= PLOT ROLLOUT =========================
plot_rollout(logs_demo)
