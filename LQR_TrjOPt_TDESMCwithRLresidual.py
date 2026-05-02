# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 01:12:30 2025

@author: elecomp
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import os, json, math
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
    m1: float = 5000.0     # [kg] Grey base mass (solid disk)
    R1: float = 3.10    # [m] Grey base radius
    m2: float = 600.5     # [kg] Blue post mass
    a2: float = 1.5    # [m] Blue post local radius
    r1: float = .5    # [m] Blue post radial offset from spin axis
    m3: float = 600.8     # [kg] Orange post mass
    a3: float = 1.5   # [m] Orange post local radius
    r2: float = 0.5    # [m] Orange post radial offset from spin axis
    m4: float = 300.4     # [kg] Red link mass (slender bar)
    L4: float = 1.20    # [m] Red link length
    l4_c: float = 0.60  # [m] Red link COM distance from hinge along the link
    gamma: float = 0.0  # [rad] Red link COM azimuth w.r.t orange post
    beta2: float = 0.0  # [rad] Blue post COM azimuth about spin axis
    beta4: float = 0.0  # [rad] Red link COM azimuth about spin axis
    alpha: float = math.radians(5.0)  # [rad] Platform roll tilt
    phi: float = math.radians(3.0)    # [rad] Platform pitch tilt
    g: float = 9.81                   # [m/s²] Gravity magnitude
    subtract_gravity_in_ueq: bool = False  # Gravity handled by TDE when False
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
    q_theta: float = 50.0      # angle error weight
    q_omega: float = 5.0       # speed weight
    r_u: float = 0.01          # control effort weight
    qT_theta: float = 2000.0   # terminal angle weight
    qT_omega: float = 50.0     # terminal speed weight
    omega_limit_penalty: float = 1000.0  # extra penalty for |omega| beyond limit

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
    lambda_s: float = 30.0
    k: float = 0.6
    phi: float = 0.02

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
# Cost configuration for RL reward shaping
# -------------------------

@dataclass
class CostConfig:
    """Weights for the per-step cost used as negative reward in RL."""
    w_e: float = 5.0        # angle error weight
    w_edot: float = 0.5     # velocity error weight
    w_u: float = 0.02       # total control effort weight (includes RL residual)
    w_omega: float = 0.2    # absolute speed penalty
    goal_tol: float = 1e-2  # termination tolerance for |θ-θ_goal| and |ω|
    done_bonus: float = 3.0 # extra reward when finished early

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
# RL Agent API (minimal interface that both SIMPLE and SAC implement)
# -------------------------

class ResidualAgentAPI:
    """Abstract interface the main loop expects for any residual RL agent."""
    def act(self, obs: Dict[str, float], eval: bool = False) -> float:
        raise NotImplementedError
    def begin_episode(self):
        pass
    def end_episode(self):
        pass
    def observe(self, o, a, r, o2, d):
        """Store one transition for training (SAC uses it; SIMPLE uses internal buffers)."""
        pass
    def update(self):
        """Perform one learning update step (SAC uses per-step; SIMPLE updates at episode end)."""
        pass
    def save(self, name: str, out_dir: str = "agents"):
        raise NotImplementedError

# -------------------------
# Rollout (training or evaluation) with any residual agent
# -------------------------

def rollout_once(
    plant: OneDOFRotorPlant,
    nom: NominalModel,
    task: Task,
    lqr_w: LQRWeights,
    smc_cfg: SMCConfig,
    agent: Optional[ResidualAgentAPI],
    cost_cfg: CostConfig,
    reference: Optional[Dict[str, object]] = None,
    seed: int = 0,
    collect_logs: bool = False,
):
    """Run one episode. If agent is None, runs SMC without residual.

    Parameters
    ----------
    reference : Optional[Dict[str, object]]
        When provided, overrides the default iLQR trajectory with a custom
        reference profile.  Supported options include:
          - ``{"kind": "constant"}`` to hold the goal angle constant.
          - ``{"kind": "piecewise_quad", "tm_frac": 0.4}`` for the
            piecewise quadratic profile shown in the user snippet.
          - ``{"theta": array_like}`` to directly specify the angle sequence
            (length ``N`` or ``N+1``) used as the reference.

    Returns (metrics, logs)
      metrics: dict with total_cost, finished (0/1), time
      logs (if collect_logs=True): dict of arrays with keys:
        't', 'theta_ref', 'omega_ref', 'alpha_ref', 'theta', 'omega', 'tau_m',
        'u_rl', 'u_eq', 'u_s', 's', 'eta_hat', 'd_hat', 'u_tde', 'u_smc',
        'u_total', 'disturbance', 'gravity', 'reference_kind'
    """
    np.random.seed(seed)
    dt = plant.p.dt
    horizon_s = time_horizon(task.theta0, task.theta_goal)
    N = int(round(horizon_s / dt))

    # Reference trajectory: optimized iLQR by default, optional custom profiles otherwise
    ref_opts = {} if reference is None else dict(reference)
    if 'duration' in ref_opts:
        horizon_s = float(ref_opts['duration'])
        N = int(round(horizon_s / dt))
    ref_kind_input = ref_opts.get('kind', 'ilqr')
    ref_kind_lower = str(ref_kind_input).lower() if ref_kind_input is not None else 'ilqr'
    has_theta_sequence = 'theta' in ref_opts

    use_ilqr_reference = (not has_theta_sequence) and (ref_kind_lower in ('ilqr', 'optimized', 'lqr'))

    if N > 0:
        t_nodes = np.arange(N + 1, dtype=float) * dt
        T = t_nodes[-1]
    else:
        t_nodes = np.zeros(1, dtype=float)
        T = 0.0

    if use_ilqr_reference:
        reference_kind = 'ilqr'
        x0 = np.array([task.theta0, task.omega0], dtype=float)
        xg = np.array([task.theta_goal, 0.0], dtype=float)
        x_ref, _ = generate_reference_ilqr_like(nom, plant.p, x0, xg, N=N, dt=dt, w=lqr_w)
        theta_ref = np.concatenate([[task.theta0], x_ref[:-1, 0]])
    else:
        if has_theta_sequence:
            reference_kind = str(ref_kind_input).lower() if ref_kind_input is not None else 'manual'
            if reference_kind in ('ilqr', 'optimized', 'lqr'):
                reference_kind = 'manual'
            theta_seq = np.asarray(ref_opts['theta'], dtype=float)
            if theta_seq.ndim != 1:
                raise ValueError("Custom theta trajectory must be a 1-D array")
            if len(theta_seq) == N + 1:
                theta_profile = theta_seq.copy()
            elif len(theta_seq) == N:
                theta_profile = np.empty(N + 1, dtype=float)
                theta_profile[0] = task.theta0
                theta_profile[1:] = theta_seq
            else:
                raise ValueError(
                    f"Custom theta trajectory length {len(theta_seq)} does not match expected {N} or {N+1} steps"
                )
        else:
            reference_kind = ref_kind_lower
            theta0 = float(ref_opts.get('theta0', task.theta0))
            theta_goal = float(ref_opts.get('theta_goal', task.theta_goal))
            if N == 0:
                theta_profile = np.array([theta0], dtype=float)
            elif ref_kind_lower == 'constant':
                target = float(ref_opts.get('theta_goal', theta_goal))
                theta_profile = np.full(N + 1, target, dtype=float)
                theta_profile[0] = theta0
            elif ref_kind_lower in ('piecewise_quad', 'pwq', 'pquad'):
                tm_frac = float(ref_opts.get('tm_frac', 0.5))
                tm_frac = min(max(tm_frac, 1e-3), 1.0 - 1e-3)
                t_m = tm_frac * T
                delta_T = max(T - t_m, 1e-12)
                a = (theta_goal - theta0) / (t_m * (t_m + delta_T)) if T > 0 else 0.0
                b = -(a * t_m) / delta_T if T > 0 else 0.0
                theta_profile = np.empty_like(t_nodes)
                for idx, tk in enumerate(t_nodes):
                    if tk <= t_m:
                        theta_profile[idx] = theta0 + a * tk * tk
                    else:
                        tau = tk - t_m
                        theta_m = theta0 + a * t_m * t_m
                        v_m = 2.0 * a * t_m
                        theta_profile[idx] = theta_m + v_m * tau + b * tau * tau
            else:
                raise ValueError(f"Unknown reference kind '{ref_kind_input}'")
        if theta_profile[0] != task.theta0:
            theta_profile = theta_profile.copy()
            theta_profile[0] = task.theta0
        theta_ref = theta_profile

    theta_ref = np.asarray(theta_ref, dtype=float)
    omega_ref = finite_diff(theta_ref, dt)
    alpha_ref = finite_diff(omega_ref, dt)

    smc = TDE_SMC_Discrete(
        hatJ=nom.J,
        hatb=nom.b,
        dt=plant.p.dt,
        u_max=plant.p.u_max,
        tau_i=plant.p.tau_i,
        K_t=plant.p.K_t,
        smc=smc_cfg,
    )
    smc.subtract_gravity_in_ueq = plant.p.subtract_gravity_in_ueq
    smc.reset()
    plant.reset(theta0=task.theta0, omega0=task.omega0)

    # Logs
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
    t = 0.0
    steps_taken = 0
    u_rl_limit = 0.2 * plant.p.u_max

    if agent is not None:
        agent.begin_episode()

    for k in range(N):
        theta, omega, tau_m = plant.state.copy()
        theta_ref_k = theta_ref[k]
        theta_ref_k1 = theta_ref[min(k + 1, N)]
        omega_ref_k = omega_ref[k]
        omega_ref_k1 = omega_ref[min(k + 1, N)]
        obs = dict(
            e=theta - theta_ref_k,
            edot=omega - omega_ref_k,
            s=(omega - omega_ref_k) + smc_cfg.lambda_s * (theta - theta_ref_k),
            omega=omega,
            tau_m=tau_m,
            time_frac=float(k) / max(1, N - 1),
        )
        o = np.array([obs['e'], obs['edot'], obs['s'], obs['omega'], 1.0, obs['time_frac']], dtype=np.float32)
        o = np.clip(o, -5.0, 5.0)

        if agent is None:
            u_rl = 0.0
        else:
            u_rl = float(agent.act(obs, eval=False))
            u_rl = sat(u_rl, u_rl_limit)

        u_cmd, info = smc.control(
            theta,
            omega,
            theta_ref_k,
            theta_ref_k1,
            omega_ref_k,
            omega_ref_k1,
            u_rl,
            plant=plant,
        )
        plant.step(u_cmd)

        e = info['e']
        edot = info['edot']
        stage = (
            cost_cfg.w_e * e * e
            + cost_cfg.w_edot * edot * edot
            + cost_cfg.w_omega * omega * omega
            + cost_cfg.w_u * u_rl * u_rl
        )
        r = -stage * dt
        total_cost += stage * dt

        theta2, omega2, _ = plant.state.copy()
        next_idx = min(k + 1, N - 1)
        obs2 = dict(
            e=theta2 - theta_ref[next_idx],
            edot=omega2 - omega_ref[next_idx],
            s=(omega2 - omega_ref[next_idx]) + smc_cfg.lambda_s * (theta2 - theta_ref[next_idx]),
            omega=omega2,
            tau_m=plant.last_tau_m,
            time_frac=float(next_idx) / max(1, N - 1),
        )
        o2 = np.array([obs2['e'], obs2['edot'], obs2['s'], obs2['omega'], 1.0, obs2['time_frac']], dtype=np.float32)
        o2 = np.clip(o2, -5.0, 5.0)

        if (abs(theta2 - task.theta_goal) < cost_cfg.goal_tol) and (abs(omega2) < cost_cfg.goal_tol):
            done = True

        if agent is not None:
            agent.observe(o, np.array([u_rl], dtype=np.float32), r, o2, float(done))
            agent.update()

        if collect_logs:
            t_log[k] = t
            th_ref_log[k] = theta_ref_k
            om_ref_log[k] = omega_ref_k
            al_ref_log[k] = alpha_ref[k]
            th_log[k] = theta
            om_log[k] = omega
            tau_m_log[k] = tau_m
            u_rl_log[k] = u_rl
            u_eq_log[k] = info['u_eq']
            u_s_log[k] = info['u_s']
            s_log[k] = info['s']
            eta_hat_log[k] = info['eta_hat']
            d_hat_log[k] = info['d_hat']
            u_tde_log[k] = -info['d_hat']
            u_smc_log[k] = info['u_smc']
            u_total_log[k] = info['u_total']
            dist_log[k] = plant.last_disturbance
            gravity_log[k] = plant.last_gravity

        steps_taken = k + 1
        if done:
            break
        t += dt

    if agent is not None:
        agent.end_episode()

    metrics = dict(total_cost=total_cost, finished=1.0 if done else 0.0, time=t)
    logs = None
    if collect_logs:
        steps = steps_taken
        logs = dict(
            t=t_log[:steps],
            theta_ref=th_ref_log[:steps],
            omega_ref=om_ref_log[:steps],
            alpha_ref=al_ref_log[:steps],
            theta=th_log[:steps],
            omega=om_log[:steps],
            tau_m=tau_m_log[:steps],
            u_rl=u_rl_log[:steps],
            u_eq=u_eq_log[:steps],
            u_s=u_s_log[:steps],
            s=s_log[:steps],
            eta_hat=eta_hat_log[:steps],
            d_hat=d_hat_log[:steps],
            u_tde=u_tde_log[:steps],
            u_smc=u_smc_log[:steps],
            u_total=u_total_log[:steps],
            disturbance=dist_log[:steps],
            gravity=gravity_log[:steps],
            reference_kind=reference_kind,
        )
    return metrics, logs

# -------------------------
# Agent factory, saving and loading
# -------------------------

AGENTS_DIR = "agents"
META_EXT = ".meta.json"

# Import residual-control RL agents if available.
# They are not needed for trj_type={"LQR", "NN", "RL"} trajectory selection,
# but the old residual-agent training helpers still use them.
try:
    from rl_simple import SimpleResidualPolicy, RLConfig as SimpleRLConfig
except Exception:
    class SimpleRLConfig:
        def __init__(self, *args, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    class SimpleResidualPolicy(ResidualAgentAPI):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("rl_simple.py is missing; SIMPLE residual-control training is unavailable.")

try:
    from rl_sac import SACResidualPolicy, SACConfig
except Exception:
    class SACConfig:
        def __init__(self, *args, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    class SACResidualPolicy(ResidualAgentAPI):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("rl_sac.py is missing; SAC residual-control training is unavailable.")


def save_meta(name: str, meta: Dict):
    os.makedirs(AGENTS_DIR, exist_ok=True)
    with open(os.path.join(AGENTS_DIR, name + META_EXT), 'w') as f:
        json.dump(meta, f, indent=2)

def load_meta(name: str) -> Dict:
    with open(os.path.join(AGENTS_DIR, name + META_EXT), 'r') as f:
        return json.load(f)


def make_agent(agent_type: str, u_rl_max: float, hidden_sizes: Tuple[int, ...] = (32, 32)) -> ResidualAgentAPI:
    """Factory: create an untrained residual agent of given type.
    agent_type: 'simple' or 'sac'
    u_rl_max: residual torque bound (magnitude)
    hidden_sizes: network layer sizes for SAC agent
    """
    if agent_type == 'simple':
        cfg = SimpleRLConfig(u_rl_max=u_rl_max)
        return SimpleResidualPolicy(n_features=6, cfg=cfg)
    elif agent_type == 'sac':
        cfg = SACConfig(u_rl_max=u_rl_max, hidden_sizes=hidden_sizes)
        return SACResidualPolicy(obs_dim=6, act_dim=1, cfg=cfg)
    else:
        raise ValueError("agent_type must be 'simple' or 'sac'")

# -------------------------
# Training & saving
# -------------------------

def train_and_save(
    agent_name: str,
    agent_type: str,
    plant_p: PlantParams,
    nom: NominalModel,
    lqr_w: LQRWeights,
    smc_cfg: SMCConfig,
    task: Task,
    cost_cfg: CostConfig,
    total_steps: int = 60000,
    start_random_steps: int = 2000,
    seed: int = 0,
):
    """Train chosen agent type on the specified task, then save it under `agent_name`.

    For 'simple': steps are interpreted as episodes*steps_per_episode internally.
    For 'sac'    : off-policy training with a replay buffer until `total_steps` samples.
    """
    np.random.seed(seed)
    plant = OneDOFRotorPlant(plant_p)

    # Residual policy limited to 20% of the SMC torque bounds
    u_rl_max = 0.2 * plant_p.u_max

    if agent_type == 'simple':
        agent = make_agent('simple', u_rl_max=u_rl_max)
        dt = plant.p.dt
        min_goal, max_goal = -abs(task.theta_goal), abs(task.theta_goal)
        print(f"Training SIMPLE residual RL for up to {total_steps} steps...")
        steps_done = 0
        ep = 0
        while steps_done < total_steps:
            ep += 1
            goal = np.random.uniform(min_goal, max_goal)
            random_task = Task(theta0=task.theta0, omega0=task.omega0, theta_goal=goal)
            horizon_s = time_horizon(random_task.theta0, random_task.theta_goal)
            steps_per_ep = int(round(horizon_s / dt))
            _, _ = rollout_once(plant, nom, random_task, lqr_w, smc_cfg, agent, cost_cfg, seed=seed+ep, collect_logs=False)
            steps_done += steps_per_ep
            if ep % 10 == 0:
                print(f"Ep {ep}")
        agent.save(agent_name, out_dir=AGENTS_DIR)
        save_meta(agent_name, {"type": "simple", "u_rl_max": agent.cfg.u_rl_max})
        print(f"Saved SIMPLE agent as '{agent_name}' in ./{AGENTS_DIR}")

    elif agent_type == 'sac':
        agent = make_agent('sac', u_rl_max=u_rl_max)
        dt = plant.p.dt
        min_goal, max_goal = -abs(task.theta_goal), abs(task.theta_goal)
        print(f"Training SAC residual RL until {total_steps} transitions...")
        ep = 0
        while getattr(agent, 'replay').len < total_steps:
            ep += 1
            goal = np.random.uniform(min_goal, max_goal)
            random_task = Task(theta0=task.theta0, omega0=task.omega0, theta_goal=goal)
            horizon_s = time_horizon(random_task.theta0, random_task.theta_goal)
            metrics, _ = rollout_once(plant, nom, random_task, lqr_w, smc_cfg, agent, cost_cfg, seed=seed+ep, collect_logs=False)
            if ep % 5 == 0:
                buf_len = getattr(agent, 'replay').len
                print(f"Ep {ep:03d}: cost={metrics['total_cost']:.4f}, finished={int(metrics['finished'])}, buffer={buf_len}")
        agent.save(agent_name, out_dir=AGENTS_DIR)
        save_meta(agent_name, {"type": "sac", "u_rl_max": agent.cfg.u_rl_max, "hidden_sizes": list(agent.cfg.hidden_sizes)})
        print(f"Saved SAC agent as '{agent_name}' in ./{AGENTS_DIR}")
    else:
        raise ValueError("agent_type must be 'simple' or 'sac'")


# -------------------------
# Trajectory-type reference generation (LQR / NN / RL)
# -------------------------

def _torch_import():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError(
            "trj_type='NN' or trj_type='RL' needs PyTorch to load saved weights. "
            f"Import failed: {exc}"
        )
    return torch, nn


def _safe_trapz(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(y) < 2:
        return 0.0
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def _safe_cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(y, dtype=float)
    if len(y) >= 2:
        out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * np.diff(x))
    return out


def _minimum_jerk(z: np.ndarray) -> np.ndarray:
    return 10.0 * z**3 - 15.0 * z**4 + 6.0 * z**5


def _nn_case_input(theta_goal: float, alpha: float, phi: float) -> np.ndarray:
    max_tilt = math.radians(20.0)
    return np.array([theta_goal / math.pi, alpha / max_tilt, phi / max_tilt], dtype=np.float32)


def _nn_shape_basis(z: np.ndarray, n_shape: int) -> np.ndarray:
    return np.stack([np.sin(k * math.pi * z) for k in range(1, n_shape + 1)], axis=1)


def _softplus_np(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _build_nn_monotone_reference(
    theta_goal: float,
    params: np.ndarray,
    dt: float,
    n_shape: int = 5,
    omega_ref_limit: float = 0.2,
    min_duration: float = 0.5,
    max_duration: float = 80.0,
    theta0: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """Same inference-time trajectory decoder as trajectory_rl_gps_addon.py."""
    params = np.asarray(params, dtype=float).reshape(-1)
    if params.size != n_shape + 1:
        raise ValueError(f"NN params must have length {n_shape + 1}, got {params.size}")

    delta = float(theta_goal - theta0)
    if abs(delta) < 1e-12:
        t = np.arange(0.0, min_duration + 0.5 * dt, dt)
        return np.full_like(t, theta0, dtype=float), float(t[-1])

    z_grid = np.linspace(0.0, 1.0, 1001)
    B = _nn_shape_basis(z_grid, n_shape)
    logits = B @ params[:n_shape]
    endpoint_factor = z_grid * (1.0 - z_grid)
    v_shape = endpoint_factor * np.exp(np.clip(logits, -5.0, 5.0)) + 1e-6
    area = _safe_trapz(v_shape, z_grid)
    v_norm = v_shape / max(area, 1e-12)
    h_grid = _safe_cumtrapz(v_norm, z_grid)
    h_grid /= max(h_grid[-1], 1e-12)

    max_hprime = float(np.max(np.abs(v_norm)))
    T_min_vel = abs(delta) * max_hprime / max(omega_ref_limit, 1e-12)
    duration_slack = float(_softplus_np(np.array([params[-1]]))[0])
    T = float(np.clip(T_min_vel + duration_slack, min_duration, max_duration))

    N = max(1, int(round(T / dt)))
    t = np.arange(N + 1, dtype=float) * dt
    T = float(t[-1])
    z = np.clip(t / max(T, 1e-12), 0.0, 1.0)
    h = np.interp(z, z_grid, h_grid)
    theta_ref = theta0 + delta * h
    theta_ref[0] = theta0
    theta_ref[-1] = theta_goal
    return theta_ref, T


def _load_nn_reference(theta_goal: float, alpha: float, phi: float, dt: float, nn_dir: str, theta0: float = 0.0) -> Tuple[np.ndarray, float]:
    """Load trajectory_policy.pt without importing trajectory_rl_gps_addon.py."""
    torch, nn = _torch_import()
    path = os.path.join(nn_dir, "trajectory_policy.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"NN trajectory weights not found: {path}. Run trajectory_rl_gps_addon.py first, "
            "or pass nn_dir pointing to a folder containing trajectory_policy.pt."
        )

    n_shape = 5
    hidden_sizes = (64, 64)
    net = nn.Sequential(
        nn.Linear(3, hidden_sizes[0]), nn.Tanh(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.Tanh(),
        nn.Linear(hidden_sizes[1], n_shape + 1),
    )
    state = torch.load(path, map_location="cpu")
    net.load_state_dict(state)
    net.eval()
    x = torch.as_tensor(_nn_case_input(theta_goal, alpha, phi), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        params = net(x).squeeze(0).cpu().numpy().astype(float)
    params[:-1] = np.clip(params[:-1], -4.0, 4.0)
    params[-1] = np.clip(params[-1], -4.0, 12.0)
    return _build_nn_monotone_reference(theta_goal, params, dt, n_shape=n_shape, theta0=theta0)


def _lqr_guide_shape_for_rl(theta_ref: np.ndarray, theta_goal: float, n_grid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_old = np.linspace(0.0, 1.0, len(theta_ref))
    z_grid = np.linspace(0.0, 1.0, n_grid)
    raw = np.asarray(theta_ref, dtype=float) - float(theta_ref[0])
    scale = float(raw[-1]) if len(raw) else 0.0
    if abs(theta_goal) < 1e-12 or abs(scale) < 1e-9:
        h = _minimum_jerk(z_grid) if abs(theta_goal) >= 1e-12 else np.zeros_like(z_grid)
    else:
        h_old = np.maximum.accumulate(np.clip(raw / scale, 0.0, 1.0))
        h_old[0] = 0.0
        h_old[-1] = 1.0
        h = np.interp(z_grid, z_old, h_old)
        h = np.maximum.accumulate(np.clip(0.85 * h + 0.15 * _minimum_jerk(z_grid), 0.0, 1.0))
        h[0] = 0.0
        h[-1] = 1.0
    v = np.maximum(np.gradient(h, z_grid), 1e-6)
    cap = max(5.0, 3.0 * float(np.percentile(v, 95)))
    v = np.minimum(v, cap)
    v /= max(_safe_trapz(v, z_grid), 1e-12)
    return z_grid, h, v


def _rl_residual_basis(z: np.ndarray, n_basis: int) -> np.ndarray:
    return np.stack([np.sin(k * math.pi * z) for k in range(1, n_basis + 1)], axis=1)


def _decode_rl_duration(raw_duration: float, theta_goal: float, T_guide: float) -> float:
    omega_ref_limit = 0.2
    duration_margin = 1.05
    max_duration_factor = 2.5
    min_extra_duration = 5.0
    duration_exp_scale = 0.35
    delta = abs(theta_goal)
    if delta < 1e-12:
        return max(0.5, T_guide)
    T_min = duration_margin * delta / omega_ref_limit
    T_center = max(T_guide, T_min)
    T_max = max(max_duration_factor * T_min, T_min + min_extra_duration, T_center)
    return float(np.clip(T_center * math.exp(duration_exp_scale * float(raw_duration)), T_min, T_max))


def _build_rl_guided_reference(
    theta_goal: float,
    params: np.ndarray,
    base_theta_ref: np.ndarray,
    T_base: float,
    dt: float,
    theta0: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """Inference-time decoder used by the SAC-GPS actor: LQR guide + learned residual."""
    params = np.asarray(params, dtype=float).reshape(-1)
    n_basis = 6
    z_grid_size = 1201
    duration_margin = 1.05
    omega_ref_limit = 0.2
    if params.size != n_basis + 1:
        raise ValueError(f"RL params must have length {n_basis + 1}, got {params.size}")
    delta = float(theta_goal - theta0)
    if abs(delta) < 1e-12:
        T = max(0.5, T_base)
        N = max(1, int(math.ceil(T / dt)))
        return np.full(N + 1, theta0, dtype=float), N * dt

    z, _, v_base = _lqr_guide_shape_for_rl(base_theta_ref, theta_goal, z_grid_size)
    B = _rl_residual_basis(z, n_basis)
    residual_log = np.clip(B @ params[:n_basis], -3.0, 3.0)
    v_gps = np.maximum(v_base, 1e-8) * np.exp(residual_log)
    v_gps = np.maximum(v_gps, 1e-9)
    v_gps /= max(_safe_trapz(v_gps, z), 1e-12)

    T = _decode_rl_duration(params[-1], theta_goal=theta_goal, T_guide=T_base)
    shape_peak = float(np.max(v_gps))
    T_needed = duration_margin * abs(delta) * shape_peak / omega_ref_limit
    T = max(T, T_needed)
    N = max(1, int(math.ceil(T / dt)))
    T = N * dt

    h_gps = _safe_cumtrapz(v_gps, z)
    h_gps = h_gps / max(h_gps[-1], 1e-12)
    h_gps = np.maximum.accumulate(np.clip(h_gps, 0.0, 1.0))
    h_gps[0] = 0.0
    h_gps[-1] = 1.0

    t = np.arange(N + 1, dtype=float) * dt
    h_time = np.interp(t / max(T, 1e-12), z, h_gps)
    theta_ref = theta0 + delta * h_time
    theta_ref[0] = theta0
    theta_ref[-1] = theta_goal
    return theta_ref, T


def _load_rl_reference(theta_goal: float, alpha: float, phi: float, dt: float, rl_dir: str, theta0: float = 0.0) -> Tuple[np.ndarray, float]:
    """Load SAC-GPS actor weights without importing sac_gps_lqr_guided_saved_teacher_addon.py."""
    torch, nn = _torch_import()
    checkpoint = os.path.join(rl_dir, "sac_gps_agent.pt")
    actor_only = os.path.join(rl_dir, "sac_gps_actor.pt")
    path = checkpoint if os.path.exists(checkpoint) else actor_only
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"RL actor weights not found in {rl_dir}. Expected sac_gps_agent.pt or sac_gps_actor.pt. "
            "Run sac_gps_lqr_guided_saved_teacher_addon.py first."
        )

    obs_dim = 3
    act_dim = 7
    hidden_sizes = (128, 128)
    net = nn.Sequential(
        nn.Linear(obs_dim, hidden_sizes[0]), nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
        nn.Linear(hidden_sizes[1], 2 * act_dim),
    )
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("actor", ckpt) if isinstance(ckpt, dict) else ckpt
    # SAC actor stores the sequential network under the prefix 'net.net'.
    if any(k.startswith("net.net.") for k in state.keys()):
        translated = {k.replace("net.net.", "", 1): v for k, v in state.items() if k.startswith("net.net.")}
    else:
        translated = state
    net.load_state_dict(translated)
    net.eval()

    obs = np.array([theta_goal / math.pi, alpha / math.radians(20.0), phi / math.radians(20.0)], dtype=np.float32)
    x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        mu_logstd = net(x)
        mu, _ = torch.chunk(mu_logstd, 2, dim=-1)
        action = torch.tanh(mu).squeeze(0).cpu().numpy().astype(np.float32)
    scales = np.r_[np.ones(6) * 1.30, 1.20].astype(np.float32)
    params = np.clip(action, -1.0, 1.0) * scales

    # Build the LQR guide with this main file's nominal model, then apply the RL residual.
    T_base = time_horizon(theta0, theta_goal)
    N_base = max(1, int(round(T_base / dt)))
    x0 = np.array([theta0, 0.0], dtype=float)
    xg = np.array([theta_goal, 0.0], dtype=float)
    # Use the same default LQR weights as the current rollout call through the public wrapper below.
    raise RuntimeError("Internal error: _load_rl_reference must be called through build_trajectory_reference so lqr_w/nom/plant_p are available.")


def build_trajectory_reference(
    trj_type: str,
    theta0: float,
    theta_goal: float,
    plant_p: PlantParams,
    nom: NominalModel,
    lqr_w: LQRWeights,
    nn_dir: str = "trajectory_rl_results",
    rl_dir: str = "true_gps_results_smoke",
) -> Optional[Dict[str, object]]:
    """Return a rollout_once reference dictionary for trj_type in {'LQR','NN','RL'}.

    LQR returns None so rollout_once uses its original built-in LQR generator.
    NN loads trajectory_policy.pt and decodes a learned monotone reference.
    RL loads the saved SAC-GPS actor and decodes an LQR-guided residual reference.
    No training is done here.
    """
    kind = str(trj_type).upper()
    if kind == "LQR":
        return None
    if kind == "NN":
        theta_ref, T = _load_nn_reference(theta_goal, plant_p.alpha, plant_p.phi, plant_p.dt, nn_dir, theta0=theta0)
        return {"kind": "NN", "theta": theta_ref, "duration": T}
    if kind == "RL":
        # Reuse the actor-loading part, but build the LQR guide using the current main-file model.
        torch, nn = _torch_import()
        checkpoint = os.path.join(rl_dir, "sac_gps_agent.pt")
        actor_only = os.path.join(rl_dir, "sac_gps_actor.pt")
        path = checkpoint if os.path.exists(checkpoint) else actor_only
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"RL actor weights not found in {rl_dir}. Expected sac_gps_agent.pt or sac_gps_actor.pt."
            )
        obs_dim, act_dim, hidden_sizes = 3, 7, (128, 128)
        net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]), nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
            nn.Linear(hidden_sizes[1], 2 * act_dim),
        )
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("actor", ckpt) if isinstance(ckpt, dict) else ckpt
        translated = {k.replace("net.net.", "", 1): v for k, v in state.items() if k.startswith("net.net.")} if any(k.startswith("net.net.") for k in state.keys()) else state
        net.load_state_dict(translated)
        net.eval()
        obs = np.array([theta_goal / math.pi, plant_p.alpha / math.radians(20.0), plant_p.phi / math.radians(20.0)], dtype=np.float32)
        with torch.no_grad():
            mu_logstd = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            mu, _ = torch.chunk(mu_logstd, 2, dim=-1)
            action = torch.tanh(mu).squeeze(0).cpu().numpy().astype(np.float32)
        params = np.clip(action, -1.0, 1.0) * np.r_[np.ones(6) * 1.30, 1.20].astype(np.float32)

        T_base = time_horizon(theta0, theta_goal)
        N_base = max(1, int(round(T_base / plant_p.dt)))
        x_ref, _ = generate_reference_ilqr_like(
            nom, plant_p,
            np.array([theta0, 0.0], dtype=float),
            np.array([theta_goal, 0.0], dtype=float),
            N=N_base,
            dt=plant_p.dt,
            w=lqr_w,
        )
        base_ref = np.concatenate([[theta0], x_ref[:-1, 0]])
        theta_ref, T = _build_rl_guided_reference(theta_goal, params, base_ref, T_base, plant_p.dt, theta0=theta0)
        return {"kind": "RL", "theta": theta_ref, "duration": T}
    raise ValueError("trj_type must be one of 'LQR', 'NN', or 'RL'")

# -------------------------
# Evaluation / Rollout API as requested
# -------------------------

def evaluate_and_rollout(
    agent_name: str,
    theta0: float,
    theta_goal: float,
    plant_p: PlantParams,
    nom: NominalModel,
    lqr_w: LQRWeights,
    smc_cfg: SMCConfig,
    cost_cfg: CostConfig,
    reference: Optional[Dict[str, object]] = None,
    seed: int = 123,
    trj_type: str = "LQR",
    nn_dir: str = "trajectory_rl_results",
    rl_dir: str = "true_gps_results_smoke",
) -> Dict[str, np.ndarray]:
    """Simulate one rollout using a saved agent or pure TDE SMC.

    If ``agent_name`` is ``'none'`` (case-insensitive) the rollout uses the
    TDE SMC controller without any residual RL action.  Otherwise the function
    loads the specified agent and includes its residual control.

    ``reference`` is forwarded to :func:`rollout_once` to allow custom
    trajectories for both the pure TDE+SMC and RL-augmented controllers.

    Returns a dictionary with arrays (each length ~ steps):
      - 't' : time [s]
      - 'theta_ref', 'omega_ref', 'alpha_ref'
      - 'theta', 'omega', 'tau_m'       (actual plant)
      - 'u_eq', 'u_s', 's', 'eta_hat', 'd_hat', 'u_tde', 'u_smc'
      - 'u_rl'                          (RL residual)
      - 'u_total', 'disturbance', 'gravity'        (command, total disturbance, gravity torque)
    """
    agent: Optional[ResidualAgentAPI] = None
    if agent_name is not None and agent_name.lower() != 'none':
        meta = load_meta(agent_name)
        a_type = meta.get('type')

        # Recreate agent and load weights
        if a_type == 'simple':
            agent = make_agent('simple', u_rl_max=meta.get('u_rl_max', plant_p.u_max * 0.2))
            agent.load(agent_name, in_dir=AGENTS_DIR)
        elif a_type == 'sac':
            hs = tuple(meta.get('hidden_sizes', (32, 32)))
            agent = make_agent('sac', u_rl_max=meta.get('u_rl_max', plant_p.u_max * 0.2), hidden_sizes=hs)
            agent.load(agent_name, in_dir=AGENTS_DIR)
        else:
            raise ValueError(f"Unknown agent type in meta: {a_type}")

    # If the caller did not pass a custom reference, generate it from trj_type.
    # For trj_type="LQR" this remains None, preserving the original behavior.
    if reference is None:
        reference = build_trajectory_reference(
            trj_type=trj_type,
            theta0=theta0,
            theta_goal=theta_goal,
            plant_p=plant_p,
            nom=nom,
            lqr_w=lqr_w,
            nn_dir=nn_dir,
            rl_dir=rl_dir,
        )

    task = Task(theta0=theta0, omega0=0.0, theta_goal=theta_goal)
    plant = OneDOFRotorPlant(plant_p)
    metrics, logs = rollout_once(
        plant, nom, task, lqr_w, smc_cfg, agent, cost_cfg, reference=reference, seed=seed, collect_logs=True
    )
    logs['metrics'] = metrics
    logs['trj_type'] = str(trj_type).upper() if reference is None else str(reference.get('kind', trj_type)).upper()
    return logs


def plot_rollout_and_errors(
    agent_name: str, theta0: float, theta_goal: float, reference: Optional[Dict[str, object]] = None
) -> None:
    """Plot trajectories and errors for an agent or pure TDE SMC.

    If ``agent_name`` is ``'none'`` the function plots only the TDE SMC
    trajectory.  Otherwise it compares the agent-augmented controller against a
    pure TDE SMC baseline.  In both cases the initial and goal angles are
    indicated and the total cost is printed.  The optional ``reference``
    argument is passed through to :func:`evaluate_and_rollout` so the same
    custom trajectory can be reused for both controllers.  Run the helper twice
    (with and without ``reference``) to compare a custom profile against the
    optimized iLQR result.
    """
    import matplotlib.pyplot as plt

    plant_p, nom, lqr_w, smc_cfg, cost_cfg = default_params()

    # Pure TDE SMC only
    if agent_name is None or agent_name.lower() == 'none':
        logs_smc = evaluate_and_rollout(
            agent_name='none',
            theta0=theta0,
            theta_goal=theta_goal,
            plant_p=plant_p,
            nom=nom,
            lqr_w=lqr_w,
            smc_cfg=smc_cfg,
            cost_cfg=cost_cfg,
            reference=reference,
        )
        print(f"TDE SMC total cost: {logs_smc['metrics']['total_cost']:.3f}")

        plt.figure()
        plt.plot(logs_smc['t'], logs_smc['theta'], label='SMC-only θ')
        plt.plot(logs_smc['t'], logs_smc['theta_ref'], '--', label='Reference θ')
        plt.axhline(theta0, color='k', linestyle=':', label='θ₀')
        plt.axhline(theta_goal, color='k', linestyle='-.', label='θ goal')
        plt.xlabel('Time [s]')
        plt.ylabel('θ [rad]')
        plt.legend()
        plt.tight_layout()

        plt.figure()
        plt.plot(
            logs_smc['t'],
            logs_smc['theta'] - logs_smc['theta_ref'],
            label='SMC error',
        )
        plt.xlabel('Time [s]')
        plt.ylabel('Tracking error [rad]')
        plt.legend()
        plt.tight_layout()

        plt.figure()
        plt.plot(
            logs_smc['t'],
            logs_smc['omega'] - logs_smc['omega_ref'],
            label='SMC ω error',
        )
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity error [rad/s]')
        plt.legend()
        plt.tight_layout()

        plt.show()
        return

    # Agent vs. TDE SMC comparison
    logs_agent = evaluate_and_rollout(
        agent_name=agent_name,
        theta0=theta0,
        theta_goal=theta_goal,
        plant_p=plant_p,
        nom=nom,
        lqr_w=lqr_w,
        smc_cfg=smc_cfg,
        cost_cfg=cost_cfg,
        reference=reference,
    )
    print(f"Agent total cost: {logs_agent['metrics']['total_cost']:.3f}")

    plant = OneDOFRotorPlant(plant_p)
    task = Task(theta0=theta0, omega0=0.0, theta_goal=theta_goal)
    metrics_smc, logs_smc = rollout_once(
        plant, nom, task, lqr_w, smc_cfg, agent=None, cost_cfg=cost_cfg, reference=reference, collect_logs=True
    )
    print(f"TDE SMC total cost: {metrics_smc['total_cost']:.3f}")

    plt.figure()
    plt.plot(logs_agent['t'], logs_agent['theta'], label='RL+SMC θ')
    plt.plot(logs_agent['t'], logs_agent['theta_ref'], '--', label='Reference θ')
    plt.plot(logs_smc['t'], logs_smc['theta'], label='SMC-only θ')
    plt.axhline(theta0, color='k', linestyle=':', label='θ₀')
    plt.axhline(theta_goal, color='k', linestyle='-.', label='θ goal')
    plt.xlabel('Time [s]')
    plt.ylabel('θ [rad]')
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(
        logs_agent['t'],
        logs_agent['theta'] - logs_agent['theta_ref'],
        label='Agent error',
    )
    plt.plot(
        logs_smc['t'],
        logs_smc['theta'] - logs_smc['theta_ref'],
        label='SMC error',
    )
    plt.xlabel('Time [s]')
    plt.ylabel('Tracking error [rad]')
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(
        logs_agent['t'],
        logs_agent['omega'] - logs_agent['omega_ref'],
        label='Agent ω error',
    )
    plt.plot(
        logs_smc['t'],
        logs_smc['omega'] - logs_smc['omega_ref'],
        label='SMC ω error',
    )
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity error [rad/s]')
    plt.legend()
    plt.tight_layout()

    plt.show()

#%% -------------------------
# Default parameter presets (you can tweak in one place)
# -------------------------

def default_params():
    plant_p = PlantParams(
        J=14099,
        b=0.09,
        u_max= 500,
        omega_max=8.0,
        dt=0.002,
        tau_i=0.1,
        K_t=5.0,
        m1 = 5000.0,     # [kg] Grey base mass (solid disk)
        R1 = 3.10  ,  # [m] Grey base radius
        m2  = 600.5,     # [kg] Blue post mass
        a2  = 1.5  ,  # [m] Blue post local radius
        r1  = .5   , # [m] Blue post radial offset from spin axis
        m3  = 600.8,     # [kg] Orange post mass
        a3  = 1.5  , # [m] Orange post local radius
        r2  = 0.5  ,  # [m] Orange post radial offset from spin axis
        m4  = 300.4,     # [kg] Red link mass (slender bar)
        L4  = 1.20 ,   # [m] Red link length
        l4_c  = 0.60,  # [m] Red link COM distance from hinge along the link
        gamma  = 0.0,  # [rad] Red link COM azimuth w.r.t orange post
        beta2  = 0.0,  # [rad] Blue post COM azimuth about spin axis
        beta4  = 0.0,  # [rad] Red link COM azimuth about spin axis
        alpha  = math.radians(25.0),  # [rad] Platform roll tilt
        phi  = math.radians(3.0)  ,  # [rad] Platform pitch tilt
        g  = 9.81                 ,  # [m/s²] Gravity magnitude
        subtract_gravity_in_ueq = False  # Gravity handled by TDE when False

    )
    
    nom = NominalModel(J=13000, b=0.06)
    lqr_w = LQRWeights(q_theta=85.0, q_omega=18.0, r_u=0.02, qT_theta=4200.0, qT_omega=220.0)
    smc_cfg = SMCConfig(lambda_s=35.0, k=0.85, phi=0.025)
    cost_cfg = CostConfig(w_e=8.0, w_edot=1.0, w_u=0.03, w_omega=0.3, goal_tol=1e-2, done_bonus=2.0)
    return plant_p, nom, lqr_w, smc_cfg, cost_cfg


#%%
if __name__ == "__main__":
    #%%
    import matplotlib.pyplot as plt

    plant_p, nom, lqr_w, smc_cfg, cost_cfg = default_params()
    theta0 = 0.0
    theta_goal = math.radians(-90.0)
    demo_duration = 150.2

    logs_demo = evaluate_and_rollout(
        agent_name="none",
        theta0=theta0,
        theta_goal=theta_goal,
        plant_p=plant_p,
        nom=nom,
        lqr_w=lqr_w,
        smc_cfg=smc_cfg,
        cost_cfg=cost_cfg,
        reference={"duration": demo_duration},
        seed=0,
    )

    t = logs_demo['t']
    theta = logs_demo['theta']
    theta_ref = logs_demo['theta_ref']
    omega = logs_demo['omega']
    omega_ref = logs_demo['omega_ref']
    u_total = logs_demo['u_total']
    tau_m = logs_demo['tau_m']
    s = logs_demo['s']
    d_hat = logs_demo['d_hat']

    J_u = float(np.sum(u_total ** 2) * plant_p.dt)
    final_error = float(theta[-1] - theta_goal)
    print(f"Demo energy metric J_u = {J_u:.4f} N·m²·s")
    print(f"Final angle error = {final_error:.6f} rad ({math.degrees(final_error):.3f} deg)")

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 10))
    fig.suptitle("Discrete TDE+SMC demo: 0 → 90° step")

    axes[0].plot(t, theta, label='θ')
    axes[0].plot(t, theta_ref, '--', label='θ_ref')
    axes[0].set_ylabel('θ [rad]')
    axes[0].legend(loc='best')

    axes[1].plot(t, omega, label='ω')
    axes[1].plot(t, omega_ref, '--', label='ω_ref')
    axes[1].set_ylabel('ω [rad/s]')
    axes[1].legend(loc='best')

    axes[2].plot(t, u_total, label='u command')
    axes[2].plot(t, tau_m, label='τ_m (shaft)')
    axes[2].set_ylabel('Torque [N·m]')
    axes[2].legend(loc='best')

    axes[3].plot(t, s, label='s (sliding)')
    axes[3].plot(t, d_hat, label='d̂ (disturbance estimate)')
    axes[3].set_ylabel('SMC/TDE')
    axes[3].set_xlabel('Time [s]')
    axes[3].legend(loc='best')

    for ax in axes:
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    #%% 1) Train and save a SIMPLE agent
    train_and_save(agent_name="demo_simple", agent_type='simple',
                  plant_p=plant_p, nom=nom, lqr_w=lqr_w, smc_cfg=smc_cfg,
                  task=Task(theta0=theta0, omega0=0.0, theta_goal=math.radians(180)),
                  cost_cfg=cost_cfg, total_steps=30000)

    #%% 2) Train and save a SAC agent
    train_and_save(agent_name="demo_sac", agent_type='sac',
                  plant_p=plant_p, nom=nom, lqr_w=lqr_w, smc_cfg=smc_cfg,
                  task=Task(theta0=0.0, omega0=0.0, theta_goal=math.radians(180)),
                  cost_cfg=cost_cfg, total_steps=60000)

    #%% 3) Evaluate a saved agent and get all logs with ifferent reference trajectory

    agent_to_evaluate = "demo_sac"  # change to the agent you saved or set to 'none'
    custom_reference = {"kind": "piecewise_quad", "tm_frac": 0.4} # "kind": "piecewise_quad" or "constant"
    comparison_setups = [
        ("Constant reference", custom_reference), # "Piecewise quadratic" or "Constant reference"
        ("Optimized iLQR", None),
    ]

    evaluation_cases = []
    for label, reference in comparison_setups:
        case = {"label": label, "reference": reference}
        logs_smc = evaluate_and_rollout(
            agent_name="none",
            theta0=theta0,
            theta_goal=theta_goal,
            plant_p=plant_p,
            nom=nom,
            lqr_w=lqr_w,
            smc_cfg=smc_cfg,
            cost_cfg=cost_cfg,
            reference=reference,
        )
        case["smc"] = logs_smc
        print(f"[{label}] TDE SMC total cost: {logs_smc['metrics']['total_cost']:.3f}")
        smc_mean_torque = float(np.mean(logs_smc['u_total']))
        print(f"[{label}] Mean torque (TDE SMC): {smc_mean_torque:.4f} Nm")

        if agent_to_evaluate and agent_to_evaluate.lower() != "none":
            try:
                logs_agent = evaluate_and_rollout(
                    agent_name=agent_to_evaluate,
                    theta0=theta0,
                    theta_goal=theta_goal,
                    plant_p=plant_p,
                    nom=nom,
                    lqr_w=lqr_w,
                    smc_cfg=smc_cfg,
                    cost_cfg=cost_cfg,
                    reference=reference,
                )
            except FileNotFoundError as exc:
                print(f"[{label}] Skipping agent '{agent_to_evaluate}': {exc}")
                logs_agent = None
                agent_to_evaluate = "none"
            else:
                case["agent"] = logs_agent
                print(f"[{label}] RL+SMC total cost: {logs_agent['metrics']['total_cost']:.3f}")
                agent_mean_torque = float(np.mean(logs_agent['u_total']))
                print(f"[{label}] Mean torque (RL+SMC): {agent_mean_torque:.4f} Nm")

        evaluation_cases.append(case)

    for case in evaluation_cases:
        label = case["label"]
        reference = case["reference"]
        logs_smc = case["smc"]
        logs_agent = case.get("agent")

        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 11))
        ref_title = f"{label} reference"
        if reference:
            ref_title += f" ({reference})"
        fig.suptitle(ref_title)

        axes[0].plot(logs_smc['t'], logs_smc['theta_ref'], 'k--', linewidth=1.1, label='Reference θ')
        axes[0].plot(logs_smc['t'], logs_smc['theta'], label='TDE+SMC θ')
        if logs_agent is not None:
            axes[0].plot(logs_agent['t'], logs_agent['theta'], label='RL+SMC θ')
        axes[0].set_ylabel('θ [rad]')
        axes[0].legend(loc='best')

        axes[1].plot(
            logs_smc['t'],
            logs_smc['theta'] - logs_smc['theta_ref'],
            label='TDE+SMC θ error',
        )
        if logs_agent is not None:
            axes[1].plot(
                logs_agent['t'],
                logs_agent['theta'] - logs_agent['theta_ref'],
                label='RL+SMC θ error',
            )
        axes[1].set_ylabel('θ error [rad]')
        axes[1].legend(loc='best')

        axes[2].plot(logs_smc['t'], logs_smc['u_smc'], label='TDE+SMC u_smc')
        if np.any(np.abs(logs_smc['u_rl']) > 1e-9):
            axes[2].plot(logs_smc['t'], logs_smc['u_rl'], label='TDE+SMC u_rl')
        if logs_agent is not None:
            axes[2].plot(logs_agent['t'], logs_agent['u_smc'], label='RL+SMC u_smc')
            axes[2].plot(logs_agent['t'], logs_agent['u_rl'], label='RL residual')
        axes[2].set_ylabel('Input effort [Nm]')
        axes[2].legend(loc='best')

        axes[3].plot(logs_smc['t'], logs_smc['u_total'], label='TDE+SMC torque')
        if logs_agent is not None:
            axes[3].plot(logs_agent['t'], logs_agent['u_total'], label='RL+SMC torque')
        axes[3].set_ylabel('Torque [Nm]')
        axes[3].set_xlabel('Time [s]')
        axes[3].legend(loc='best')

        plt.figure()
        plt.plot(logs_smc['t'], logs_smc['omega_ref'], 'k--', linewidth=1.1, label='Reference ω')
        plt.plot(logs_smc['t'], logs_smc['omega'], label='TDE+SMC ω')
        if logs_agent is not None:
            plt.plot(logs_agent['t'], logs_agent['omega'], label='RL+SMC ω')
        plt.xlabel('Time [s]')
        plt.ylabel('ω [rad/s]')
        plt.legend(loc='best')

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

