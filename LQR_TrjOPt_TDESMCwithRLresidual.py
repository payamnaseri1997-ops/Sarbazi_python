# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 14:49:17 2025

@author: elecomp
"""

# rotor_ilqr_smc_rl.py
# 1-DoF rotational system:
# Optimal trajectory (finite-horizon LQR with saturation) + TDE-SMC tracking + residual RL adaptation.

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, List
import math
import numpy as np

# -------------------------
# Utilities
# -------------------------

def sat(x: float, limit: float) -> float:
    return max(-limit, min(limit, x))

def sat_vec(x: np.ndarray, limit: float) -> np.ndarray:
    return np.clip(x, -limit, limit)

def finite_diff(y: np.ndarray, dt: float) -> np.ndarray:
    """Centered difference for interior, forward/backward for ends."""
    dy = np.zeros_like(y)
    if len(y) >= 2:
        dy[0] = (y[1] - y[0]) / dt
        dy[-1] = (y[-1] - y[-2]) / dt
    if len(y) >= 3:
        dy[1:-1] = (y[2:] - y[:-2]) / (2.0 * dt)
    return dy

# -------------------------
# Plant and model
# -------------------------

@dataclass
class PlantParams:
    J: float          # inertia (true)
    b: float          # viscous damping (true)
    u_max: float      # torque limit
    omega_max: float  # speed soft-limit (used in cost, not hard clamp here)
    dt: float         # control/integ step [s]

@dataclass
class NominalModel:
    J: float
    b: float

class OneDOFRotorPlant:
    """
    True plant: J * domega = u - b*omega + d(t)
    d(t) includes Coulomb friction, periodic load, and noise.
    """
    def __init__(self, p: PlantParams):
        self.p = p
        self.state = np.zeros(2)  # [theta, omega]
        # Disturbance params
        self.torque_coulomb = 0.02  # Nm
        self.load_amp = 0.03        # Nm
        self.load_freq = 1.5        # Hz
        self.noise_std = 0.01       # Nm
        self.time = 0.0

    def reset(self, theta0: float = 0.0, omega0: float = 0.0) -> np.ndarray:
        self.state[:] = [theta0, omega0]
        self.time = 0.0
        return self.state.copy()

    def step(self, u_cmd: float) -> np.ndarray:
        u = sat(u_cmd, self.p.u_max)
        theta, omega = self.state
        # Disturbances
        d_coul = self.torque_coulomb * (1.0 if omega >= 0 else -1.0) if abs(omega) > 1e-5 else 0.0
        d_per = self.load_amp * math.sin(2.0 * math.pi * self.load_freq * self.time)
        d_noise = np.random.randn() * self.noise_std
        d = d_coul + d_per + d_noise

        # Dynamics integration (semi-implicit Euler)
        domega = (u - self.p.b * omega + d) / self.p.J
        omega_next = omega + domega * self.p.dt
        theta_next = theta + omega_next * self.p.dt  # update with new omega (semi-implicit)
        self.state[:] = [theta_next, omega_next]
        self.time += self.p.dt
        return self.state.copy()

# -------------------------
# iLQR-style finite-horizon LQR (with torque clamp) to generate reference
# -------------------------

@dataclass
class LQRWeights:
    q_theta: float = 50.0      # angle error weight
    q_omega: float = 5.0       # speed weight (also keeps |omega| small)
    r_u: float = 0.01          # control effort
    qT_theta: float = 2000.0   # terminal angle weight
    qT_omega: float = 50.0     # terminal speed weight

def build_AB(nom: NominalModel, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discrete-time linear model x_{k+1} = A x_k + B u_k, x = [theta, omega]
    From: theta_dot = omega; omega_dot = (u - b*omega)/J
    """
    J, b = nom.J, nom.b
    A = np.array([[1.0, dt],
                  [0.0, 1.0 - (b/J)*dt]])
    B = np.array([[0.0],
                  [dt / J]])
    return A, B

def finite_horizon_lqr_gain(A: np.ndarray, B: np.ndarray, N: int, w: LQRWeights) -> List[np.ndarray]:
    """
    Backward Riccati to get time-varying gains K[k].
    """
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
    """
    Use finite-horizon LQR feedback with torque saturation to roll out a feasible reference (x_ref, u_ref).
    Soft state constraints (e.g., |omega|) are encouraged via Q weights; torque is hard-clamped.
    """
    A, B = build_AB(nom, dt)
    Ks = finite_horizon_lqr_gain(A, B, N, w)

    x_ref = np.zeros((N + 1, 2))
    u_ref = np.zeros(N)

    x = x0.copy()
    for k in range(N):
        x_tilde = x - x_goal
        u = float(-Ks[k] @ x_tilde.reshape(2, 1))  # unconstrained
        u = sat(u, plant_limits.u_max)             # enforce torque limit
        # Simulate nominal discrete model for reference
        x = A @ x + B.flatten() * u
        x_ref[k, :] = x
        u_ref[k] = u
    x_ref[N, :] = x
    return x_ref, u_ref

# -------------------------
# Time Delay Estimator (TDE) + Sliding Mode Controller (SMC)
# -------------------------

@dataclass
class SMCConfig:
    lambda_s: float = 30.0     # sliding surface slope
    k: float = 0.6             # sliding gain
    phi: float = 0.02          # boundary layer for sat(s/phi)
    delay_steps: int = 2       # TDE delay steps (>=1)

class TDE_SMC_Controller:
    """
    u = u_eq - d_hat - k * sat(s / phi) + u_RL
    with s = e_dot + lambda_s * e
    TDE: d_hat(t) = J_nom * omega_dot(t-Δ) - u(t-Δ) + b_nom * omega(t-Δ)
         omega_dot(t-Δ) ≈ (omega(t) - omega(t-Δ)) / (Δ*dt)
    """
    def __init__(self, nom: NominalModel, limits: PlantParams, smc: SMCConfig):
        self.nom = nom
        self.limits = limits
        self.cfg = smc
        self._omega_hist: List[float] = []
        self._u_hist: List[float] = []

    def reset(self):
        self._omega_hist.clear()
        self._u_hist.clear()

    def control(self,
                theta: float, omega: float,
                theta_ref: float, omega_ref: float, alpha_ref: float,
                u_rl: float) -> float:
        e = theta - theta_ref
        edot = omega - omega_ref
        s = edot + self.cfg.lambda_s * e

        # Equivalent control (unknown dynamics replaced by nominal)
        u_eq = self.nom.J * (alpha_ref - self.cfg.lambda_s * edot) + self.nom.b * omega

        # TDE estimate
        self._omega_hist.append(omega)
        self._u_hist.append(0.0)  # placeholder; will be overwritten after u_total computed

        d_hat = 0.0
        m = self.cfg.delay_steps
        if len(self._omega_hist) > m:
            omega_now = self._omega_hist[-1]
            omega_del = self._omega_hist[-1 - m]
            u_del = self._u_hist[-1 - m]
            omega_dot_del = (omega_now - omega_del) / (m * self.limits.dt)
            d_hat = self.nom.J * omega_dot_del - u_del + self.nom.b * omega_del

        # Sliding action with boundary layer
        s_norm = s / (self.cfg.phi + 1e-9)
        u_s = -self.cfg.k * np.tanh(s_norm)  # smooth sat to avoid chattering

        u_total = u_eq - d_hat + u_s + u_rl
        u_total = sat(u_total, self.limits.u_max)

        # write actual applied u into history
        self._u_hist[-1] = u_total
        return u_total, dict(e=e, edot=edot, s=s, d_hat=d_hat, u_eq=u_eq, u_s=u_s)

# -------------------------
# Residual RL policy (REINFORCE)
# -------------------------

@dataclass
class RLConfig:
    lr: float = 1e-3
    sigma: float = 0.1      # exploration std in action space (Nm)
    gamma: float = 0.995
    u_rl_max: float = 0.15  # cap residual torque magnitude
    feature_clip: float = 5.0

class ResidualPolicy:
    """
    Gaussian policy: a_rl ~ N(mu= w^T phi(s), sigma^2)
    Trained with episodic REINFORCE.
    """
    def __init__(self, n_features: int, cfg: RLConfig):
        self.w = np.zeros(n_features)  # linear policy weights
        self.cfg = cfg

        # Buffers for one episode
        self.phi_traj: List[np.ndarray] = []
        self.mu_traj: List[float] = []
        self.a_traj: List[float] = []
        self.r_traj: List[float] = []

    def features(self, obs: Dict[str, float]) -> np.ndarray:
        # obs keys: e, edot, s, omega, time_frac
        phi = np.array([
            obs["e"], obs["edot"], obs["s"], obs["omega"], 1.0, obs["time_frac"]
        ], dtype=float)
        return sat_vec(phi, self.cfg.feature_clip)

    def act(self, obs: Dict[str, float]) -> float:
        phi = self.features(obs)
        mu = float(np.dot(self.w, phi))
        # bound mean softly
        mu = float(sat(mu, self.cfg.u_rl_max))
        a = np.random.randn() * self.cfg.sigma + mu
        a = float(sat(a, self.cfg.u_rl_max))
        # log (approx) probability for Gaussian with clipped sample (ignore squash correction for simplicity)
        self.phi_traj.append(phi)
        self.mu_traj.append(mu)
        self.a_traj.append(a)
        return a

    def step_reward(self, r: float):
        self.r_traj.append(r)

    def finish_and_update(self):
        # compute returns
        R = 0.0
        G = np.zeros(len(self.r_traj))
        for t in reversed(range(len(self.r_traj))):
            R = self.r_traj[t] + self.cfg.gamma * R
            G[t] = R
        # simple baseline: mean return
        b = np.mean(G) if len(G) > 0 else 0.0

        # gradient update
        grad = np.zeros_like(self.w)
        var = self.cfg.sigma ** 2
        for phi, mu, a, g in zip(self.phi_traj, self.mu_traj, self.a_traj, G):
            advantage = g - b
            # ∇_w log π ≈ ((a - mu)/var) * ∂mu/∂w  with mu = w^T phi
            grad += ((a - mu) / var) * phi * advantage

        self.w += self.cfg.lr * grad

        # clear buffers
        self.phi_traj.clear()
        self.mu_traj.clear()
        self.a_traj.clear()
        self.r_traj.clear()

# -------------------------
# Training / Simulation
# -------------------------

@dataclass
class Task:
    theta0: float
    omega0: float
    theta_goal: float
    horizon_s: float

@dataclass
class CostConfig:
    w_e: float = 5.0       # angle error
    w_edot: float = 0.5    # velocity error
    w_u: float = 0.02      # total control effort
    w_omega: float = 0.2   # absolute speed
    goal_tol: float = 1e-2
    done_bonus: float = 3.0

def simulate_episode(
    plant: OneDOFRotorPlant,
    nom: NominalModel,
    task: Task,
    lqr_w: LQRWeights,
    smc_cfg: SMCConfig,
    rl_policy: ResidualPolicy | None,
    cost_cfg: CostConfig,
    seed: int = 0
) -> Dict[str, float]:
    np.random.seed(seed)

    # discretization
    dt = plant.p.dt
    N = int(round(task.horizon_s / dt))

    # reference via iLQR-like LQR rollout (on nominal model, torque-clamped)
    x0 = np.array([task.theta0, task.omega0], dtype=float)
    xg = np.array([task.theta_goal, 0.0], dtype=float)
    x_ref, u_ref = generate_reference_ilqr_like(
        nom, plant.p, x0, xg, N=N, dt=dt, w=lqr_w
    )
    theta_ref = np.concatenate([[task.theta0], x_ref[:-1, 0]])  # shift so ref[0] = initial
    omega_ref = finite_diff(theta_ref, dt)
    alpha_ref = finite_diff(omega_ref, dt)

    # controller
    smc = TDE_SMC_Controller(nom, plant.p, smc_cfg)
    smc.reset()

    plant.reset(theta0=task.theta0, omega0=task.omega0)

    total_cost = 0.0
    done = False
    t = 0.0
    for k in range(N):
        theta, omega = plant.state.copy()
        # RL residual
        if rl_policy is not None:
            obs = dict(
                e=theta - theta_ref[k],
                edot=omega - omega_ref[k],
                s=(omega - omega_ref[k]) + smc_cfg.lambda_s * (theta - theta_ref[k]),
                omega=omega,
                time_frac=float(k) / max(1, N - 1),
            )
            u_rl = rl_policy.act(obs)
        else:
            u_rl = 0.0

        u_cmd, info = smc.control(
            theta=theta, omega=omega,
            theta_ref=theta_ref[k], omega_ref=omega_ref[k], alpha_ref=alpha_ref[k],
            u_rl=u_rl
        )
        plant.step(u_cmd)

        # stage cost (negative reward)
        e = info["e"]
        edot = info["edot"]
        stage = (cost_cfg.w_e * e * e
                 + cost_cfg.w_edot * edot * edot
                 + cost_cfg.w_omega * omega * omega
                 + cost_cfg.w_u * u_cmd * u_cmd)
        total_cost += stage * dt

        if rl_policy is not None:
            rl_policy.step_reward(-stage * dt)  # reward is negative cost

        # early stopping if within goal tolerance and nearly stopped
        if (abs(theta - task.theta_goal) < cost_cfg.goal_tol) and (abs(omega) < cost_cfg.goal_tol):
            done = True
            if rl_policy is not None:
                rl_policy.step_reward(cost_cfg.done_bonus)  # encourage finishing early
            break
        t += dt

    if rl_policy is not None:
        rl_policy.finish_and_update()

    return dict(total_cost=total_cost, finished=1.0 if done else 0.0, time=t)

# -------------------------
# Entry point: quick training loop
# -------------------------

def train_residual_rl():
    # True plant vs nominal model (mismatch on J, b)
    plant_p = PlantParams(J=0.045, b=0.09, u_max=0.8, omega_max=8.0, dt=0.002)
    nom = NominalModel(J=0.05, b=0.06)

    plant = OneDOFRotorPlant(plant_p)

    # LQR weights for reference generation (soft speed constraint via q_omega)
    lqr_w = LQRWeights(q_theta=80.0, q_omega=15.0, r_u=0.02, qT_theta=4000.0, qT_omega=200.0)

    # SMC and RL configs
    smc_cfg = SMCConfig(lambda_s=40.0, k=0.8, phi=0.03, delay_steps=3)
    rl_cfg = RLConfig(lr=1e-3, sigma=0.08, gamma=0.997, u_rl_max=0.18)
    policy = ResidualPolicy(n_features=6, cfg=rl_cfg)

    task = Task(theta0=0.0, omega0=0.0, theta_goal=math.radians(90.0), horizon_s=1.2)
    cost_cfg = CostConfig(w_e=8.0, w_edot=1.0, w_u=0.03, w_omega=0.3, goal_tol=1e-2, done_bonus=2.0)

    # Train for a handful of episodes (increase for better results)
    n_episodes = 30
    print("Training residual RL on top of TDE-SMC tracking an LQR reference...")
    for ep in range(1, n_episodes + 1):
        metrics = simulate_episode(
            plant, nom, task, lqr_w, smc_cfg, policy, cost_cfg, seed=ep
        )
        print(f"Ep {ep:03d}: cost={metrics['total_cost']:.4f}, "
              f"time={metrics['time']:.3f}s, finished={int(metrics['finished'])}")

    # Baseline (no RL) for comparison
    print("\nEvaluating baseline controller (no residual RL):")
    base_metrics = simulate_episode(
        plant, nom, task, lqr_w, smc_cfg, rl_policy=None, cost_cfg=cost_cfg, seed=999
    )
    print(f"Baseline: cost={base_metrics['total_cost']:.4f}, "
          f"time={base_metrics['time']:.3f}s, finished={int(base_metrics['finished'])}")

if __name__ == "__main__":
    train_residual_rl()
