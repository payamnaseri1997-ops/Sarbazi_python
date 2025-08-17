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
    """True plant parameters used inside the simulator.
    J: inertia [kg·m^2]
    b: viscous damping [N·m·s/rad]
    u_max: absolute torque limit [N·m]
    omega_max: soft speed limit used in cost function [rad/s]
    dt: control / integration time step [s]
    """
    J: float
    b: float
    u_max: float
    omega_max: float
    dt: float

@dataclass
class NominalModel:
    """Nominal model parameters used in controller/trajectory computations."""
    J: float
    b: float

class OneDOFRotorPlant:
    """True 1-DoF second-order rotational system with disturbances.

    State: x = [theta, omega]
    Dynamics: J * domega = u - b*omega + d(t)
    Disturbance d(t) includes Coulomb friction, periodic load, an additional
    sinusoid with time-varying amplitude and frequency (to mimic unmodeled
    dynamics), and white torque noise.
    """
    def __init__(self, p: PlantParams):
        self.p = p
        self.state = np.zeros(2)  # [theta, omega]
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

    def reset(self, theta0: float = 0.0, omega0: float = 0.0) -> np.ndarray:
        """Reset the plant state.
        theta0: initial angle [rad]
        omega0: initial angular velocity [rad/s]
        Returns the current state copy.
        """
        self.state[:] = [theta0, omega0]
        self.time = 0.0
        return self.state.copy()

    def step(self, u_cmd: float) -> np.ndarray:
        """Advance dynamics one time step with applied torque u_cmd.
        Applies saturation to respect u_max and integrates with semi-implicit Euler.
        Returns the updated state copy.
        """
        u = sat(u_cmd, self.p.u_max)
        theta, omega = self.state
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
        d = d_coul + d_per + d_var + d_noise

        # Dynamics integration (semi-implicit Euler)
        domega = (u - self.p.b * omega + d) / self.p.J
        omega_next = omega + domega * self.p.dt
        theta_next = theta + omega_next * self.p.dt
        self.state[:] = [theta_next, omega_next]
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
    - Speed/other soft constraints are encoded through Q/Qf.
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
        u = float(-Ks[k] @ x_tilde.reshape(2, 1))
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
    delay_steps: integer delay in steps for TDE derivative/backshift.
    """
    lambda_s: float = 30.0
    k: float = 0.6
    phi: float = 0.02
    delay_steps: int = 2

class TDE_SMC_Controller:
    """Time-Delay Estimator augmented SMC.

    Control law: u = u_eq - d_hat + u_s + u_RL
    where
        s = (ω - ω_ref) + λ (θ - θ_ref),
        u_eq ≈ J_nom (α_ref - λ (ω - ω_ref)) + b_nom ω,
        d_hat is TDE disturbance estimate using delayed (ω, u),
        u_s = -k * tanh(s/φ) is continuous sliding action,
        u_RL is the residual torque from the RL agent.
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
                u_rl: float) -> Tuple[float, Dict[str, float]]:
        e = theta - theta_ref
        edot = omega - omega_ref
        s = edot + self.cfg.lambda_s * e

        # Equivalent control using nominal model
        u_eq = self.nom.J * (alpha_ref - self.cfg.lambda_s * edot) + self.nom.b * omega

        # TDE estimate
        self._omega_hist.append(omega)
        self._u_hist.append(0.0)  # placeholder; overwritten below
        d_hat = 0.0
        m = self.cfg.delay_steps
        if len(self._omega_hist) > m:
            omega_now = self._omega_hist[-1]
            omega_del = self._omega_hist[-1 - m]
            u_del = self._u_hist[-1 - m]
            omega_dot_del = (omega_now - omega_del) / (m * self.limits.dt)
            d_hat = self.nom.J * omega_dot_del - u_del + self.nom.b * omega_del

        # Sliding action
        s_norm = s / (self.cfg.phi + 1e-9)
        u_s = -self.cfg.k * np.tanh(s_norm)

        u_total = u_eq - d_hat + u_s + u_rl
        u_total = sat(u_total, self.limits.u_max)
        self._u_hist[-1] = u_total

        info = dict(e=e, edot=edot, s=s, d_hat=d_hat, u_eq=u_eq, u_s=u_s, u_smc=u_eq - d_hat + u_s)
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
    """Task specification: start, goal and time horizon."""
    theta0: float
    omega0: float
    theta_goal: float
    horizon_s: float

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
    seed: int = 0,
    collect_logs: bool = False,
):
    """Run one episode. If agent is None, runs SMC without residual.

    Returns (metrics, logs)
      metrics: dict with total_cost, finished (0/1), time
      logs (if collect_logs=True): dict of arrays with keys:
        't', 'theta_ref', 'omega_ref', 'alpha_ref', 'theta', 'omega',
        'u_rl', 'u_eq', 'u_s', 'd_hat', 'u_smc', 'u_total'
    """
    np.random.seed(seed)
    dt = plant.p.dt
    N = int(round(task.horizon_s / dt))

    # Reference trajectory on nominal model
    x0 = np.array([task.theta0, task.omega0], dtype=float)
    xg = np.array([task.theta_goal, 0.0], dtype=float)
    x_ref, _ = generate_reference_ilqr_like(nom, plant.p, x0, xg, N=N, dt=dt, w=lqr_w)
    theta_ref = np.concatenate([[task.theta0], x_ref[:-1, 0]])
    omega_ref = finite_diff(theta_ref, dt)
    alpha_ref = finite_diff(omega_ref, dt)

    smc = TDE_SMC_Controller(nom, plant.p, smc_cfg)
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
        u_rl_log = np.zeros(N)
        u_eq_log = np.zeros(N)
        u_s_log = np.zeros(N)
        d_hat_log = np.zeros(N)
        u_smc_log = np.zeros(N)
        u_total_log = np.zeros(N)

    total_cost = 0.0
    done = False
    t = 0.0

    if agent is not None:
        agent.begin_episode()

    for k in range(N):
        theta, omega = plant.state.copy()
        obs = dict(
            e=theta - theta_ref[k],
            edot=omega - omega_ref[k],
            s=(omega - omega_ref[k]) + smc_cfg.lambda_s * (theta - theta_ref[k]),
            omega=omega,
            time_frac=float(k) / max(1, N - 1),
        )
        # feature vector for storage (SAC needs ndarray form)
        o = np.array([obs['e'], obs['edot'], obs['s'], obs['omega'], 1.0, obs['time_frac']], dtype=np.float32)
        o = np.clip(o, -5.0, 5.0)

        u_rl = 0.0 if agent is None else agent.act(obs, eval=False)
        u_cmd, info = smc.control(theta, omega, theta_ref[k], omega_ref[k], alpha_ref[k], u_rl)
        plant.step(u_cmd)

        # stage cost (negative reward)
        e = info['e']; edot = info['edot']
        stage = (cost_cfg.w_e * e * e
                 + cost_cfg.w_edot * edot * edot
                 + cost_cfg.w_omega * omega * omega
                 + cost_cfg.w_u * u_cmd * u_cmd)
        r = -stage * dt
        total_cost += stage * dt

        # next obs for agent
        theta2, omega2 = plant.state.copy()
        obs2 = dict(
            e=theta2 - theta_ref[min(k+1, N-1)],
            edot=omega2 - omega_ref[min(k+1, N-1)],
            s=(omega2 - omega_ref[min(k+1, N-1)]) + smc_cfg.lambda_s * (theta2 - theta_ref[min(k+1, N-1)]),
            omega=omega2,
            time_frac=float(min(k+1, N-1)) / max(1, N - 1),
        )
        o2 = np.array([obs2['e'], obs2['edot'], obs2['s'], obs2['omega'], 1.0, obs2['time_frac']], dtype=np.float32)
        o2 = np.clip(o2, -5.0, 5.0)

        # terminal condition
        if (abs(theta2 - task.theta_goal) < cost_cfg.goal_tol) and (abs(omega2) < cost_cfg.goal_tol):
            done = True

        if agent is not None:
            agent.observe(o, np.array([u_rl], dtype=np.float32), r, o2, float(done))
            agent.update()

        if collect_logs:
            t_log[k] = t
            th_ref_log[k] = theta_ref[k]
            om_ref_log[k] = omega_ref[k]
            al_ref_log[k] = alpha_ref[k]
            th_log[k] = theta
            om_log[k] = omega
            u_rl_log[k] = u_rl
            u_eq_log[k] = info['u_eq']
            u_s_log[k] = info['u_s']
            d_hat_log[k] = info['d_hat']
            u_smc_log[k] = info['u_smc']
            u_total_log[k] = u_cmd

        if done:
            break
        t += dt

    if agent is not None:
        agent.end_episode()

    metrics = dict(total_cost=total_cost, finished=1.0 if done else 0.0, time=t)
    logs = None
    if collect_logs:
        logs = dict(
            t=t_log[:k+1], theta_ref=th_ref_log[:k+1], omega_ref=om_ref_log[:k+1], alpha_ref=al_ref_log[:k+1],
            theta=th_log[:k+1], omega=om_log[:k+1],
            u_rl=u_rl_log[:k+1], u_eq=u_eq_log[:k+1], u_s=u_s_log[:k+1], d_hat=d_hat_log[:k+1],
            u_smc=u_smc_log[:k+1], u_total=u_total_log[:k+1]
        )
    return metrics, logs

# -------------------------
# Agent factory, saving and loading
# -------------------------

AGENTS_DIR = "agents"
META_EXT = ".meta.json"

# Import RL agents (files below in this canvas)
from rl_simple import SimpleResidualPolicy, RLConfig as SimpleRLConfig
from rl_sac import SACResidualPolicy, SACConfig


def save_meta(name: str, meta: Dict):
    os.makedirs(AGENTS_DIR, exist_ok=True)
    with open(os.path.join(AGENTS_DIR, name + META_EXT), 'w') as f:
        json.dump(meta, f, indent=2)

def load_meta(name: str) -> Dict:
    with open(os.path.join(AGENTS_DIR, name + META_EXT), 'r') as f:
        return json.load(f)


def make_agent(agent_type: str, u_rl_max: float) -> ResidualAgentAPI:
    """Factory: create an untrained residual agent of given type.
    agent_type: 'simple' or 'sac'
    u_rl_max: residual torque bound (magnitude)
    """
    if agent_type == 'simple':
        cfg = SimpleRLConfig(u_rl_max=u_rl_max)
        return SimpleResidualPolicy(n_features=6, cfg=cfg)
    elif agent_type == 'sac':
        cfg = SACConfig(u_rl_max=u_rl_max)
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
        # Use episodic training; do N episodes
        dt = plant.p.dt
        steps_per_ep = int(round(task.horizon_s / dt))
        n_episodes = max(1, total_steps // steps_per_ep)
        print(f"Training SIMPLE residual RL for {n_episodes} episodes...")
        for ep in range(1, n_episodes + 1):
            _, _ = rollout_once(plant, nom, task, lqr_w, smc_cfg, agent, cost_cfg, seed=ep, collect_logs=False)
            if ep % 10 == 0:
                print(f"Ep {ep}")
        # Save model and meta
        agent.save(agent_name, out_dir=AGENTS_DIR)
        save_meta(agent_name, {"type": "simple", "u_rl_max": agent.cfg.u_rl_max})
        print(f"Saved SIMPLE agent as '{agent_name}' in ./{AGENTS_DIR}")

    elif agent_type == 'sac':
        agent = make_agent('sac', u_rl_max=u_rl_max)
        dt = plant.p.dt
        steps_per_ep = int(round(task.horizon_s / dt))
        print(f"Training SAC residual RL until {total_steps} transitions...")
        ep = 0
        while getattr(agent, 'replay').len < total_steps:
            ep += 1
            metrics, _ = rollout_once(plant, nom, task, lqr_w, smc_cfg, agent, cost_cfg, seed=seed+ep, collect_logs=False)
            if ep % 5 == 0:
                buf_len = getattr(agent, 'replay').len
                print(f"Ep {ep:03d}: cost={metrics['total_cost']:.4f}, finished={int(metrics['finished'])}, buffer={buf_len}")
        agent.save(agent_name, out_dir=AGENTS_DIR)
        save_meta(agent_name, {"type": "sac", "u_rl_max": agent.cfg.u_rl_max})
        print(f"Saved SAC agent as '{agent_name}' in ./{AGENTS_DIR}")
    else:
        raise ValueError("agent_type must be 'simple' or 'sac'")

# -------------------------
# Evaluation / Rollout API as requested
# -------------------------

def evaluate_and_rollout(
    agent_name: str,
    theta0: float,
    theta_goal: float,
    horizon_s: float,
    plant_p: PlantParams,
    nom: NominalModel,
    lqr_w: LQRWeights,
    smc_cfg: SMCConfig,
    cost_cfg: CostConfig,
    seed: int = 123
) -> Dict[str, np.ndarray]:
    """Load saved agent by `agent_name`, simulate one rollout, and return arrays.

    Returns a dictionary with arrays (each length ~ steps):
      - 't' : time [s]
      - 'theta_ref', 'omega_ref', 'alpha_ref'
      - 'theta', 'omega'                (actual plant)
      - 'u_eq', 'd_hat', 'u_s', 'u_smc' (SMC/TDE components)
      - 'u_rl'                           (RL residual)
      - 'u_total'                        (applied torque)
    """
    meta = load_meta(agent_name)
    a_type = meta.get('type')

    # Recreate agent and load weights
    if a_type == 'simple':
        agent = make_agent('simple', u_rl_max=meta.get('u_rl_max', plant_p.u_max * 0.2))
        agent.load(agent_name, in_dir=AGENTS_DIR)
    elif a_type == 'sac':
        agent = make_agent('sac', u_rl_max=meta.get('u_rl_max', plant_p.u_max * 0.2))
        agent.load(agent_name, in_dir=AGENTS_DIR)
    else:
        raise ValueError(f"Unknown agent type in meta: {a_type}")

    task = Task(theta0=theta0, omega0=0.0, theta_goal=theta_goal, horizon_s=horizon_s)
    plant = OneDOFRotorPlant(plant_p)
    metrics, logs = rollout_once(plant, nom, task, lqr_w, smc_cfg, agent, cost_cfg, seed=seed, collect_logs=True)
    logs['metrics'] = metrics
    return logs

# -------------------------
# Default parameter presets (you can tweak in one place)
# -------------------------

def default_params():
    plant_p = PlantParams(J=0.045, b=0.09, u_max=0.8, omega_max=8.0, dt=0.002)
    nom = NominalModel(J=0.05, b=0.06)
    lqr_w = LQRWeights(q_theta=80.0, q_omega=15.0, r_u=0.02, qT_theta=4000.0, qT_omega=200.0)
    smc_cfg = SMCConfig(lambda_s=40.0, k=0.8, phi=0.03, delay_steps=3)
    cost_cfg = CostConfig(w_e=8.0, w_edot=1.0, w_u=0.03, w_omega=0.3, goal_tol=1e-2, done_bonus=2.0)
    return plant_p, nom, lqr_w, smc_cfg, cost_cfg

if __name__ == "__main__":
    # === Example usage ===
    plant_p, nom, lqr_w, smc_cfg, cost_cfg = default_params()

    # 1) Train and save a SIMPLE agent
    # train_and_save(agent_name="demo_simple", agent_type='simple',
    #               plant_p=plant_p, nom=nom, lqr_w=lqr_w, smc_cfg=smc_cfg,
    #               task=Task(theta0=0.0, omega0=0.0, theta_goal=math.radians(90), horizon_s=1.2),
    #               cost_cfg=cost_cfg, total_steps=30000)

    # 2) Train and save a SAC agent
    # train_and_save(agent_name="demo_sac", agent_type='sac',
    #               plant_p=plant_p, nom=nom, lqr_w=lqr_w, smc_cfg=smc_cfg,
    #               task=Task(theta0=0.0, omega0=0.0, theta_goal=math.radians(90), horizon_s=1.2),
    #               cost_cfg=cost_cfg, total_steps=60000)

    # 3) Evaluate a saved agent and get all logs
    # logs = evaluate_and_rollout(
    #     agent_name="demo_sac", theta0=0.0, theta_goal=math.radians(60), horizon_s=1.2,
    #     plant_p=plant_p, nom=nom, lqr_w=lqr_w, smc_cfg=smc_cfg, cost_cfg=cost_cfg)
    # print("Keys in logs:", list(logs.keys()))

