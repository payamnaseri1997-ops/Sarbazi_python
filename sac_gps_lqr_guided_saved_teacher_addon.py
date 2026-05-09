# -*- coding: utf-8 -*-
"""
sac_gps_lqr_guided_saved_teacher_addon.py

Clean SAC-GPS trajectory learner.

This file trains/loads the SAC actor, but all plant parameters, base LQR
trajectory, TDE+SMC rollout, and physical simulation are delegated to:
    true_gps_lqr_guided_addon.py -> LQR_TrjOPt_TDESMCwithRLresidual.py

The SAC/GPS reward compares closed-loop rollout objective values.  It is not
the finite-horizon LQR planning objective: the LQR baseline duration is fixed
externally, while the SAC/GPS trajectory parameterization may choose duration.

Saved model format:
    save_dir/sac_gps_agent.pt/actor.keras
    save_dir/sac_gps_agent.pt/q1.keras
    save_dir/sac_gps_agent.pt/q2.keras
    save_dir/sac_gps_agent.pt/q1_t.keras
    save_dir/sac_gps_agent.pt/q2_t.keras
    save_dir/sac_gps_agent.pt/state.json
    save_dir/sac_gps_actor.pt/actor.keras
    save_dir/training_history.json
    save_dir/configs.json
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple
import os
import math
import json
import numpy as np
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception as exc:  # pragma: no cover
    tf = None
    keras = None
    layers = None
    _TF_IMPORT_ERROR = exc
else:
    _TF_IMPORT_ERROR = None


def _require_tensorflow() -> None:
    if tf is None:
        raise RuntimeError(f"TensorFlow is required for this file. TensorFlow import failed: {_TF_IMPORT_ERROR}")


def _print_tensorflow_diagnostics() -> None:
    _require_tensorflow()
    print("TensorFlow version:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

import LQR_TrjOPt_TDESMCwithRLresidual as sysmod
import true_gps_lqr_guided_addon as gps


HOLD_FRACTION_AFTER_GOAL = gps.HOLD_FRACTION_AFTER_GOAL


#%% ========================= CONFIG =========================

@dataclass
class SACGPSConfig:
    hidden_sizes: Tuple[int, int]
    actor_lr: float
    critic_lr: float
    alpha_lr: float
    gamma: float
    tau: float
    batch_size: int
    replay_size: int
    start_random_steps: int
    updates_per_interaction: int
    alpha_init: float
    autotune_alpha: bool
    target_entropy_scale: float
    shape_action_scale: float
    duration_action_scale: float
    reward_scale: float
    reward_clip: float
    use_teacher_prefill: bool
    behavior_clone_epochs: int
    behavior_clone_lr: float
    critic_refine_steps: int
    critic_refine_lr: float
    critic_refine_l2: float


def mkdir(path: Optional[str]):
    if path:
        os.makedirs(path, exist_ok=True)


def make_default_configs():
    traj_cfg, obj_cfg = gps.make_default_configs()
    sac_cfg = SACGPSConfig(
        hidden_sizes=tuple(sysmod.RL_HIDDEN_SIZES),
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        gamma=0.0,
        tau=0.005,
        batch_size=128,
        replay_size=200000,
        start_random_steps=300,
        updates_per_interaction=1,
        alpha_init=0.2,
        autotune_alpha=True,
        target_entropy_scale=1.0,
        shape_action_scale=float(sysmod.RL_SHAPE_ACTION_SCALE),
        duration_action_scale=float(sysmod.RL_DURATION_ACTION_SCALE),
        reward_scale=10000.0,
        reward_clip=10.0,
        use_teacher_prefill=True,
        behavior_clone_epochs=300,
        behavior_clone_lr=1e-3,
        critic_refine_steps=50,
        critic_refine_lr=0.03,
        critic_refine_l2=0.002,
    )
    return traj_cfg, obj_cfg, sac_cfg


def action_scales(traj_cfg: gps.GPSTrajectoryConfig, cfg: SACGPSConfig) -> np.ndarray:
    return np.r_[np.ones(traj_cfg.n_basis) * cfg.shape_action_scale, cfg.duration_action_scale].astype(np.float32)


def action_to_params(action: np.ndarray, traj_cfg: gps.GPSTrajectoryConfig, cfg: SACGPSConfig) -> np.ndarray:
    return (np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0) * action_scales(traj_cfg, cfg)).astype(float)


def params_to_action(params: np.ndarray, traj_cfg: gps.GPSTrajectoryConfig, cfg: SACGPSConfig) -> np.ndarray:
    return np.clip(np.asarray(params, dtype=np.float32).reshape(-1) / action_scales(traj_cfg, cfg), -1.0, 1.0).astype(np.float32)


def case_obs(case: gps.GPSCase) -> np.ndarray:
    return gps.case_features(case).astype(np.float32)


#%% ========================= SAC NETWORKS =========================

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

    def sample(self, batch_size: int, device=None):
        idx = np.random.randint(0, self.len, size=int(batch_size))
        return dict(
            obs=tf.convert_to_tensor(self.obs[idx], dtype=tf.float32),
            act=tf.convert_to_tensor(self.act[idx], dtype=tf.float32),
            rew=tf.convert_to_tensor(self.rew[idx], dtype=tf.float32),
            obs2=tf.convert_to_tensor(self.obs2[idx], dtype=tf.float32),
            done=tf.convert_to_tensor(self.done[idx], dtype=tf.float32),
        )


def build_mlp(inp: int, out: int, hidden_sizes: Tuple[int, ...], activation: str = "relu", name: str = "mlp"):
    _require_tensorflow()
    inputs = keras.Input(shape=(int(inp),), dtype=tf.float32, name=f"{name}_input")
    x = inputs
    for i, h in enumerate(hidden_sizes):
        x = layers.Dense(int(h), activation=activation, name=f"{name}_hidden_{i}")(x)
    outputs = layers.Dense(int(out), activation=None, name=f"{name}_output")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name=name)


MLP = build_mlp


class SquashedGaussianActor(keras.Model if keras is not None else object):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int, ...], name: str = "squashed_gaussian_actor"):
        _require_tensorflow()
        super().__init__(name=name)
        self.act_dim = int(act_dim)
        self.net = build_mlp(obs_dim, 2 * act_dim, hidden_sizes, activation="relu", name="actor_mlp")
        self.log_std_min = -5.0
        self.log_std_max = 1.0

    def call(self, obs, training: bool = False):
        return self.net(obs, training=training)

    def forward(self, obs, deterministic: bool = False, with_logprob: bool = True, training: bool = False):
        mu_logstd = self.net(obs, training=training)
        mu, log_std = tf.split(mu_logstd, 2, axis=-1)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)
        std = tf.exp(log_std)
        z = mu if deterministic else mu + std * tf.random.normal(tf.shape(mu), dtype=mu.dtype)
        action = tf.tanh(z)
        if with_logprob:
            logp = -0.5 * (tf.square((z - mu) / (std + 1e-8)) + 2.0 * log_std + tf.math.log(2.0 * math.pi))
            logp = logp - tf.math.log(1.0 - tf.square(action) + 1e-6)
            logp = tf.reduce_sum(logp, axis=-1, keepdims=True)
        else:
            logp = None
        return action, logp, tf.tanh(mu)



class Critic(keras.Model if keras is not None else object):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int, ...], name: str = "critic"):
        _require_tensorflow()
        super().__init__(name=name)
        self.q = build_mlp(obs_dim + act_dim, 1, hidden_sizes, activation="relu", name=f"{name}_mlp")

    def call(self, inputs, training: bool = False):
        obs, act = inputs
        return self.forward(obs, act, training=training)

    def forward(self, obs, act, training: bool = False):
        return self.q(tf.concat([obs, act], axis=-1), training=training)


class SACGPSAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: SACGPSConfig):
        _require_tensorflow()
        self.cfg = cfg
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.actor = SquashedGaussianActor(obs_dim, act_dim, cfg.hidden_sizes)
        self.q1 = Critic(obs_dim, act_dim, cfg.hidden_sizes, name="q1")
        self.q2 = Critic(obs_dim, act_dim, cfg.hidden_sizes, name="q2")
        self.q1_t = Critic(obs_dim, act_dim, cfg.hidden_sizes, name="q1_t")
        self.q2_t = Critic(obs_dim, act_dim, cfg.hidden_sizes, name="q2_t")
        dummy_obs = tf.zeros((1, obs_dim), dtype=tf.float32)
        dummy_act = tf.zeros((1, act_dim), dtype=tf.float32)
        self.actor.forward(dummy_obs, deterministic=True, with_logprob=False)
        self.q1.forward(dummy_obs, dummy_act); self.q2.forward(dummy_obs, dummy_act)
        self.q1_t.forward(dummy_obs, dummy_act); self.q2_t.forward(dummy_obs, dummy_act)
        self.q1_t.set_weights(self.q1.get_weights())
        self.q2_t.set_weights(self.q2.get_weights())
        self.pi_opt = tf.keras.optimizers.Adam(learning_rate=cfg.actor_lr)
        self.q1_opt = tf.keras.optimizers.Adam(learning_rate=cfg.critic_lr)
        self.q2_opt = tf.keras.optimizers.Adam(learning_rate=cfg.critic_lr)
        self.replay = ReplayBuffer(obs_dim, act_dim, cfg.replay_size)
        self.total_interactions = 0
        self.log_alpha = tf.Variable(math.log(cfg.alpha_init), dtype=tf.float32, trainable=bool(cfg.autotune_alpha), name="log_alpha")
        self.alpha_opt = tf.keras.optimizers.Adam(learning_rate=cfg.alpha_lr) if cfg.autotune_alpha else None
        self.target_entropy = -float(act_dim) * cfg.target_entropy_scale if cfg.autotune_alpha else None

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = tf.convert_to_tensor(np.asarray(obs, dtype=np.float32)[None, :], dtype=tf.float32)
        a, _, mu = self.actor.forward(obs_t, deterministic=deterministic, with_logprob=False, training=False)
        out = mu if deterministic else a
        return out.numpy().squeeze(0).astype(np.float32)

    def update(self, n_updates: int = 1) -> Dict[str, float]:
        if self.replay.len < self.cfg.batch_size:
            return {}
        logs: Dict[str, float] = {}
        for _ in range(int(n_updates)):
            batch = self.replay.sample(self.cfg.batch_size)
            obs, act, rew, obs2, done = batch["obs"], batch["act"], batch["rew"], batch["obs2"], batch["done"]
            alpha = self.alpha
            a2, logp2, _ = self.actor.forward(obs2, deterministic=False, with_logprob=True, training=False)
            q_t = tf.minimum(self.q1_t.forward(obs2, a2, training=False), self.q2_t.forward(obs2, a2, training=False)) - alpha * logp2
            backup = tf.stop_gradient(rew + self.cfg.gamma * (1.0 - done) * q_t)
            with tf.GradientTape() as tape:
                q1_pred = self.q1.forward(obs, act, training=True)
                q1_loss = tf.reduce_mean(tf.square(q1_pred - backup))
            self.q1_opt.apply_gradients(zip(tape.gradient(q1_loss, self.q1.trainable_variables), self.q1.trainable_variables))
            with tf.GradientTape() as tape:
                q2_pred = self.q2.forward(obs, act, training=True)
                q2_loss = tf.reduce_mean(tf.square(q2_pred - backup))
            self.q2_opt.apply_gradients(zip(tape.gradient(q2_loss, self.q2.trainable_variables), self.q2.trainable_variables))
            with tf.GradientTape() as tape:
                a_pi, logp_pi, _ = self.actor.forward(obs, deterministic=False, with_logprob=True, training=True)
                q_pi = tf.minimum(self.q1.forward(obs, a_pi, training=False), self.q2.forward(obs, a_pi, training=False))
                pi_loss = tf.reduce_mean(tf.stop_gradient(alpha) * logp_pi - q_pi)
            self.pi_opt.apply_gradients(zip(tape.gradient(pi_loss, self.actor.trainable_variables), self.actor.trainable_variables))
            if self.alpha_opt is not None:
                with tf.GradientTape() as tape:
                    alpha_loss = tf.reduce_mean(-self.log_alpha * tf.stop_gradient(logp_pi + self.target_entropy))
                self.alpha_opt.apply_gradients([(tape.gradient(alpha_loss, self.log_alpha), self.log_alpha)])
            else:
                alpha_loss = tf.constant(0.0, dtype=tf.float32)
            for online, target in ((self.q1, self.q1_t), (self.q2, self.q2_t)):
                for w, wt in zip(online.weights, target.weights):
                    wt.assign((1.0 - self.cfg.tau) * wt + self.cfg.tau * w)
            logs = dict(q1_loss=float(q1_loss.numpy()), q2_loss=float(q2_loss.numpy()), pi_loss=float(pi_loss.numpy()), alpha=float(self.alpha.numpy()), alpha_loss=float(alpha_loss.numpy()))
        return logs

    def behavior_clone(self, obs_arr: np.ndarray, act_arr: np.ndarray, epochs: int, lr: float) -> List[float]:
        if epochs <= 0 or len(obs_arr) == 0:
            return []
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        obs_t = tf.convert_to_tensor(obs_arr, dtype=tf.float32)
        act_t = tf.convert_to_tensor(act_arr, dtype=tf.float32)
        losses: List[float] = []
        for ep in range(int(epochs)):
            with tf.GradientTape() as tape:
                _, _, mu = self.actor.forward(obs_t, deterministic=True, with_logprob=False, training=True)
                loss = tf.reduce_mean(tf.square(mu - act_t))
            opt.apply_gradients(zip(tape.gradient(loss, self.actor.trainable_variables), self.actor.trainable_variables))
            losses.append(float(loss.numpy()))
            if (ep + 1) % max(1, epochs // 4) == 0:
                print(f"behavior clone {ep+1}/{epochs}: loss={losses[-1]:.6g}")
        return losses

    def save(self, path: str):
        mkdir(path)
        self.actor.net.save(os.path.join(path, "actor.keras"))
        self.q1.q.save(os.path.join(path, "q1.keras"))
        self.q2.q.save(os.path.join(path, "q2.keras"))
        self.q1_t.q.save(os.path.join(path, "q1_t.keras"))
        self.q2_t.q.save(os.path.join(path, "q2_t.keras"))
        with open(os.path.join(path, "state.json"), "w") as f:
            json.dump(dict(log_alpha=float(self.log_alpha.numpy()), cfg=asdict(self.cfg), total_interactions=int(self.total_interactions), replay_resumed=False, replay_note="Replay buffer is not checkpointed; resumed SAC runs reload saved CEM teachers and start with a refilled replay buffer."), f, indent=2)

    def load(self, path: str):
        self.actor.net = keras.models.load_model(os.path.join(path, "actor.keras"))
        self.q1.q = keras.models.load_model(os.path.join(path, "q1.keras"))
        self.q2.q = keras.models.load_model(os.path.join(path, "q2.keras"))
        self.q1_t.q = keras.models.load_model(os.path.join(path, "q1_t.keras"))
        self.q2_t.q = keras.models.load_model(os.path.join(path, "q2_t.keras"))
        state_path = os.path.join(path, "state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                ckpt = json.load(f)
            if "log_alpha" in ckpt:
                self.log_alpha.assign(float(ckpt["log_alpha"]))
            self.total_interactions = int(ckpt.get("total_interactions", self.total_interactions))
        print(f"Loaded SAC-GPS agent from: {path}")


#%% ========================= ENVIRONMENT HELPERS VIA TRUE GPS/LQR =========================

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
        metrics, logs, theta_ref, T, extra, base_ref, base_T = gps.evaluate_guided_params(case, params, traj_cfg, obj_cfg, seed=seed)
        cost = float(metrics["total_cost"])
        failed = False
    except Exception:
        metrics, logs, theta_ref, T, extra, base_ref, base_T = None, None, None, None, None, existing_ref, existing_T
        cost = float(existing_metrics["total_cost"] + 1e8)
        failed = True
    raw_reward = (float(existing_metrics["total_cost"]) - cost) / max(sac_cfg.reward_scale, 1e-12)
    reward = float(np.clip(raw_reward, -sac_cfg.reward_clip, sac_cfg.reward_clip))
    return dict(case=case, action=np.asarray(action, dtype=np.float32), params=params, reward=reward, raw_reward=float(raw_reward), cost=cost, existing_cost=float(existing_metrics["total_cost"]), improvement_pct=100.0 * (float(existing_metrics["total_cost"]) - cost) / max(abs(float(existing_metrics["total_cost"])), 1e-12), metrics=metrics, logs=logs, theta_ref=theta_ref, T=T, extra=extra, existing_metrics=existing_metrics, existing_logs=existing_logs, existing_ref=existing_ref, existing_T=existing_T, base_ref=base_ref, base_T=base_T, failed=failed)


def critic_refine_action(agent: SACGPSAgent, obs: np.ndarray, action_init: np.ndarray, steps: int, lr: float, l2: float) -> np.ndarray:
    if steps <= 0:
        return np.asarray(action_init, dtype=np.float32)
    obs_t = tf.convert_to_tensor(np.asarray(obs, dtype=np.float32)[None, :], dtype=tf.float32)
    a0 = np.clip(np.asarray(action_init, dtype=np.float32), -0.999, 0.999)
    y = tf.Variable(np.arctanh(a0), dtype=tf.float32)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    a0_t = tf.convert_to_tensor(a0[None, :], dtype=tf.float32)
    for _ in range(int(steps)):
        with tf.GradientTape() as tape:
            a = tf.tanh(y)[None, :]
            q = tf.minimum(agent.q1.forward(obs_t, a, training=False), agent.q2.forward(obs_t, a, training=False))
            loss = -(tf.reduce_mean(q) - l2 * tf.reduce_mean(tf.square(a - a0_t)))
        opt.apply_gradients([(tape.gradient(loss, y), y)])
    return tf.tanh(y).numpy().astype(np.float32)


#%% ========================= TEACHERS / TRAINING =========================

def _teacher_key(goal_deg: float, alpha_deg: float, phi_deg: float) -> Tuple[float, float, float]:
    return (round(float(goal_deg), 6), round(float(alpha_deg), 6), round(float(phi_deg), 6))


def _load_teacher_summary(save_dir: Optional[str]) -> Dict[Tuple[float, float, float], Dict[str, object]]:
    """Load saved teachers from old and new JSON formats.

    Supported inputs:
      * teacher_summary.json legacy rows: goal/alpha/phi/action
      * training_teachers.json rows: theta_goal_deg/alpha_deg/phi_deg/best_params
      * teacher_summary.json new rows: theta_goal_deg/alpha_deg/phi_deg/best_action/best_params
    """
    if not save_dir:
        return {}
    candidates = [
        os.path.join(save_dir, "teacher_summary.json"),
        os.path.join(save_dir, "training_teachers.json"),
    ]
    out: Dict[Tuple[float, float, float], Dict[str, object]] = {}
    loaded_paths: List[str] = []
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            rows = json.load(f)
        if isinstance(rows, dict):
            rows = rows.get("rows", rows.get("teachers", []))
        for row0 in rows:
            row = dict(row0)
            if {"goal", "alpha", "phi"}.issubset(row):
                key = _teacher_key(row["goal"], row["alpha"], row["phi"])
            elif {"theta_goal_deg", "alpha_deg", "phi_deg"}.issubset(row):
                key = _teacher_key(row["theta_goal_deg"], row["alpha_deg"], row["phi_deg"])
            else:
                continue
            has_action = "best_action" in row or "action" in row
            has_params = "best_params" in row
            if has_action and has_params:
                row["_format"] = "action_and_params"
            elif has_action:
                row["_format"] = "action"
            elif has_params:
                row["_format"] = "params"
            else:
                continue
            out[key] = row
        loaded_paths.append(path)
    if loaded_paths:
        print(f"Loaded {len(out)} saved teacher rows from: {', '.join(loaded_paths)}")
    return out


def prefill_from_saved_teachers(agent, train_cases, traj_cfg, obj_cfg, sac_cfg, baseline_cache, save_dir, seed=0):
    teacher_map = _load_teacher_summary(save_dir)
    if not teacher_map:
        return None
    rows, bc_obs, bc_act = [], [], []
    found_cases, missing_cases = [], []
    zero_action = np.zeros(traj_cfg.n_basis + 1, dtype=np.float32)
    for i, case in enumerate(train_cases):
        obs = case_obs(case)
        zero_ev = evaluate_action(case, zero_action, traj_cfg, obj_cfg, sac_cfg, baseline_cache, seed=seed + i)
        agent.replay.store(obs, zero_action, zero_ev["reward"], obs2=obs, done=True)
        saved = teacher_map.get(_teacher_key(case.theta_goal_deg, case.alpha_deg, case.phi_deg))
        if saved is None:
            missing_cases.append(case.label())
            continue
        if "best_action" in saved:
            best_action = np.asarray(saved["best_action"], dtype=np.float32)
        elif "action" in saved:
            best_action = np.asarray(saved["action"], dtype=np.float32)
        elif "best_params" in saved:
            best_action = params_to_action(np.asarray(saved["best_params"], dtype=np.float32), traj_cfg, sac_cfg)
        else:
            missing_cases.append(case.label())
            continue
        best_ev = evaluate_action(case, best_action, traj_cfg, obj_cfg, sac_cfg, baseline_cache, seed=seed + 2000 + i)
        agent.replay.store(obs, best_action, best_ev["reward"], obs2=obs, done=True)
        bc_obs.append(obs); bc_act.append(best_action)
        rows.append(dict(case=case, zero_eval=zero_ev, best_eval=best_ev, best_action=best_action, saved_teacher=saved))
        found_cases.append(case.label())
        print(f"saved teacher {i+1}/{len(train_cases)}: {case.label()} | fixed-horizon LQR rollout objective={best_ev['existing_cost']:.6g}, teacher rollout objective={best_ev['cost']:.6g}, improvement={best_ev['improvement_pct']:.2f}%")
    print(f"Saved teacher prefill found {len(found_cases)}/{len(train_cases)} cases.")
    if found_cases:
        print("Cases with saved teachers: " + ", ".join(found_cases))
    if missing_cases:
        print("Cases missing saved teachers: " + ", ".join(missing_cases))
    bc_losses = []
    if bc_obs and sac_cfg.behavior_clone_epochs > 0:
        bc_losses = agent.behavior_clone(np.stack(bc_obs), np.stack(bc_act), sac_cfg.behavior_clone_epochs, sac_cfg.behavior_clone_lr)
    return dict(rows=rows, bc_losses=bc_losses, loaded_from="saved_teacher_json", found_cases=found_cases, missing_cases=missing_cases)


def prefill_with_gps_teachers(agent, train_cases, traj_cfg, obj_cfg, sac_cfg, baseline_cache, teacher_cem_iters, teacher_population, seed, save_dir=None):
    rows, bc_obs, bc_act = [], [], []
    zero_action = np.zeros(traj_cfg.n_basis + 1, dtype=np.float32)
    for i, case in enumerate(train_cases):
        obs = case_obs(case)
        zero_ev = evaluate_action(case, zero_action, traj_cfg, obj_cfg, sac_cfg, baseline_cache, seed=seed + i)
        agent.replay.store(obs, zero_action, zero_ev["reward"], obs2=obs, done=True)
        print(f"\n=== LQR-guided local teacher {i+1}/{len(train_cases)}: {case.label()} ===")
        teacher = gps.cem_optimize_guided_case(case=case, traj_cfg=traj_cfg, obj_cfg=obj_cfg, init_mean=np.zeros(traj_cfg.n_basis + 1), init_std=np.r_[np.ones(traj_cfg.n_basis) * 0.55, 0.35], cem_iters=teacher_cem_iters, population=teacher_population, elite_frac=0.25, seed=seed + 1000 + 31 * i)
        best_params = np.asarray(teacher["best"]["params"], dtype=np.float32)
        best_action = params_to_action(best_params, traj_cfg, sac_cfg)
        best_ev = evaluate_action(case, best_action, traj_cfg, obj_cfg, sac_cfg, baseline_cache, seed=seed + 2000 + i)
        agent.replay.store(obs, best_action, best_ev["reward"], obs2=obs, done=True)
        bc_obs.append(obs); bc_act.append(best_action)
        rows.append(dict(case=case, zero_eval=zero_ev, teacher=teacher, best_eval=best_ev, best_action=best_action, best_params=best_params))
    if bc_obs:
        bc_losses = agent.behavior_clone(np.stack(bc_obs), np.stack(bc_act), sac_cfg.behavior_clone_epochs, sac_cfg.behavior_clone_lr)
    else:
        bc_losses = []
    if save_dir:
        with open(os.path.join(save_dir, "teacher_summary.json"), "w") as f:
            json.dump([dict(goal=r["case"].theta_goal_deg, alpha=r["case"].alpha_deg, phi=r["case"].phi_deg, existing=r["best_eval"]["existing_cost"], teacher=r["best_eval"]["cost"], improvement_pct=r["best_eval"]["improvement_pct"], action=r["best_action"].tolist()) for r in rows], f, indent=2)
    return dict(rows=rows, bc_losses=bc_losses, loaded_from="fresh_cem")



def _json_load_list(path: str) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return list(data.get("rows", data.get("history", data.get("teachers", []))))
    return []


def _json_dump(path: str, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def generate_or_resume_cem_teachers(
    train_cases,
    traj_cfg,
    obj_cfg,
    sac_cfg,
    save_dir,
    cem_iters=4,
    population=20,
    elite_frac=0.25,
    seed=0,
    resume=True,
    skip_existing_cases=True,
):
    """Generate GPS/LQR-guided CEM teachers one case at a time and save after each case.

    This stage does not create or train a SAC agent. It only calls the delegated
    true_gps/LQR optimization utilities and writes resumable teacher artifacts.
    """
    mkdir(save_dir)
    summary_path = os.path.join(save_dir, "teacher_summary.json")
    history_path = os.path.join(save_dir, "cem_teacher_history.json")
    config_path = os.path.join(save_dir, "latest_cem_config.json")
    requested_cases = list(train_cases)
    requested_keys = {_teacher_key(c.theta_goal_deg, c.alpha_deg, c.phi_deg) for c in requested_cases}

    existing_rows = _json_load_list(summary_path) if resume else []
    if not resume:
        existing_rows = [
            r for r in existing_rows
            if _teacher_key(
                r.get("theta_goal_deg", r.get("goal", np.nan)),
                r.get("alpha_deg", r.get("alpha", np.nan)),
                r.get("phi_deg", r.get("phi", np.nan)),
            ) not in requested_keys
        ]
    teacher_rows = list(existing_rows)
    teacher_map = {}
    for row in teacher_rows:
        if {"theta_goal_deg", "alpha_deg", "phi_deg"}.issubset(row):
            teacher_map[_teacher_key(row["theta_goal_deg"], row["alpha_deg"], row["phi_deg"])] = row
        elif {"goal", "alpha", "phi"}.issubset(row):
            teacher_map[_teacher_key(row["goal"], row["alpha"], row["phi"])] = row

    history = _json_load_list(history_path) if resume else []
    config = dict(
        timestamp=str(np.datetime64("now")),
        requested_teacher_cases=len(requested_cases),
        cem_iters=int(cem_iters),
        population=int(population),
        elite_frac=float(elite_frac),
        seed=int(seed),
        resume=bool(resume),
        skip_existing_cases=bool(skip_existing_cases),
        save_dir=save_dir,
        traj_cfg=asdict(traj_cfg),
        obj_cfg=asdict(obj_cfg),
        sac_cfg=asdict(sac_cfg),
        cases=[_case_to_dict(c) for c in requested_cases],
    )
    _json_dump(config_path, config)

    completed_cases, skipped_cases, failed_cases = [], [], []
    print(f"Requested CEM teacher cases: {len(requested_cases)}")
    for i, case in enumerate(requested_cases):
        key = _teacher_key(case.theta_goal_deg, case.alpha_deg, case.phi_deg)
        if resume and skip_existing_cases and key in teacher_map:
            skipped_cases.append(case)
            print(f"Skipping completed CEM teacher case {i+1}/{len(requested_cases)}: {case.label()}")
            continue
        print(f"\n=== CEM teacher case {i+1}/{len(requested_cases)}: {case.label()} ===")
        try:
            baseline_cache = BaselineCache(obj_cfg)
            teacher = gps.cem_optimize_guided_case(
                case=case,
                traj_cfg=traj_cfg,
                obj_cfg=obj_cfg,
                init_mean=np.zeros(traj_cfg.n_basis + 1),
                init_std=np.r_[np.ones(traj_cfg.n_basis) * 0.55, 0.35],
                cem_iters=cem_iters,
                population=population,
                elite_frac=elite_frac,
                seed=seed + 1000 + 31 * i,
            )
            best_params = np.asarray(teacher["best"]["params"], dtype=np.float32)
            best_action = params_to_action(best_params, traj_cfg, sac_cfg)
            best_ev = evaluate_action(case, best_action, traj_cfg, obj_cfg, sac_cfg, baseline_cache, seed=seed + 2000 + i)
            row = dict(
                theta_goal_deg=float(case.theta_goal_deg),
                alpha_deg=float(case.alpha_deg),
                phi_deg=float(case.phi_deg),
                existing_cost=float(best_ev["existing_cost"]),
                teacher_cost=float(best_ev["cost"]),
                improvement_pct=float(best_ev["improvement_pct"]),
                best_params=best_params.astype(float).tolist(),
                best_action=best_action.astype(float).tolist(),
                cem_iters=int(cem_iters),
                population=int(population),
                seed=int(seed + 1000 + 31 * i),
                timestamp=str(np.datetime64("now")),
            )
            teacher_rows = [
                r for r in teacher_rows
                if _teacher_key(
                    r.get("theta_goal_deg", r.get("goal", np.nan)),
                    r.get("alpha_deg", r.get("alpha", np.nan)),
                    r.get("phi_deg", r.get("phi", np.nan)),
                ) != key
            ]
            teacher_rows.append(row)
            teacher_map[key] = row
            _json_dump(summary_path, teacher_rows)
            history_record = dict(record_type="case", status="completed", case=_case_to_dict(case), teacher_row=row)
            history.append(history_record)
            _json_dump(history_path, history)
            completed_cases.append(case)
            print(f"Saved teacher summary after case {i+1}: {summary_path}")
        except Exception as exc:
            failed_cases.append(case)
            failure = dict(
                record_type="case",
                status="failed",
                timestamp=str(np.datetime64("now")),
                case=_case_to_dict(case),
                error=repr(exc),
            )
            history.append(failure)
            _json_dump(history_path, history)
            print(f"CEM teacher failed for {case.label()}: {exc!r}. Continuing.")
    print(f"CEM teacher stage complete: requested={len(requested_cases)}, completed={len(completed_cases)}, skipped={len(skipped_cases)}, failed={len(failed_cases)}")
    print(f"Teacher summary path: {summary_path}")
    return dict(
        teacher_rows=teacher_rows,
        completed_cases=completed_cases,
        skipped_cases=skipped_cases,
        failed_cases=failed_cases,
        save_dir=save_dir,
    )


def _load_history_if_available(save_dir: Optional[str]) -> List[Dict[str, object]]:
    if not save_dir:
        return []
    path = os.path.join(save_dir, "training_history.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        hist = json.load(f)
    print(f"Loaded existing training history with {len(hist)} records from: {path}")
    return hist


def _history_path(save_dir: Optional[str]) -> Optional[str]:
    return os.path.join(save_dir, "training_history.json") if save_dir else None


def _save_history(save_dir: Optional[str], history: Sequence[Dict[str, object]]) -> None:
    path = _history_path(save_dir)
    if not path:
        return
    with open(path, "w") as f:
        json.dump(list(history), f, indent=2)


def _metadata_case_keys(records: Sequence[Dict[str, object]]) -> set:
    seen = set()
    for rec in records:
        for c in rec.get("trained_cases_this_run", []):
            seen.add(f"{float(c['theta_goal_deg']):.6f}|{float(c['alpha_deg']):.6f}|{float(c['phi_deg']):.6f}")
    return seen


def _case_key(case: gps.GPSCase) -> str:
    return f"{case.theta_goal_deg:.6f}|{case.alpha_deg:.6f}|{case.phi_deg:.6f}"


def _case_to_dict(case: gps.GPSCase) -> Dict[str, float]:
    return {"theta_goal_deg": float(case.theta_goal_deg), "alpha_deg": float(case.alpha_deg), "phi_deg": float(case.phi_deg)}


def _save_sac_artifacts(agent, save_dir, history, run_record, traj_cfg, obj_cfg, sac_cfg):
    if not save_dir:
        return
    mkdir(save_dir)
    agent_path = os.path.join(save_dir, "sac_gps_agent.pt")
    actor_path = os.path.join(save_dir, "sac_gps_actor.pt")
    agent.save(agent_path)
    mkdir(actor_path)
    agent.actor.net.save(os.path.join(actor_path, "actor.keras"))
    run_record["model_path"] = agent_path
    run_record["actor_path"] = actor_path
    with open(os.path.join(save_dir, "configs.json"), "w") as f:
        json.dump(dict(sac_cfg=asdict(sac_cfg), traj_cfg=asdict(traj_cfg), obj_cfg=asdict(obj_cfg)), f, indent=2)
    latest_path = os.path.join(save_dir, "latest_run_config.json")
    with open(latest_path, "w") as f:
        json.dump(run_record, f, indent=2)
    _save_history(save_dir, history)
    print(f"Saved SAC checkpoint to: {agent_path}")
    print(f"Saved SAC actor to: {actor_path}")
    print(f"Saved training metadata to: {latest_path}")


def train_sac_gps_agent(train_cases, traj_cfg, obj_cfg, sac_cfg, total_interactions=1000, teacher_cem_iters=4, teacher_population=20, eval_every=50, seed=0, save_dir=None, use_saved_teacher_summary=True, allow_fresh_cem_teachers=False, resume_training=False, skip_existing_cases=False, save_every=50):
    _print_tensorflow_diagnostics()
    mkdir(save_dir)
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed); np.random.seed(seed)
    agent = SACGPSAgent(obs_dim=3, act_dim=traj_cfg.n_basis + 1, cfg=sac_cfg)
    baseline_cache = BaselineCache(obj_cfg)
    history = _load_history_if_available(save_dir)
    teacher_output = None
    checkpoint_path = os.path.join(save_dir, "sac_gps_agent.pt") if save_dir else "sac_gps_agent.pt"
    agent_loaded = False

    if resume_training:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"resume_training=True but no saved SAC checkpoint found at: {checkpoint_path}")
        agent.load(checkpoint_path); agent_loaded = True
        print(f"SAC checkpoint loaded: {checkpoint_path}")
    else:
        print("Starting SAC actor/critic training from scratch.")
    interactions_before = int(agent.total_interactions)
    print(f"Total interactions before training: {interactions_before}")

    requested_cases = list(train_cases)
    # CEM skip_existing_cases only controls teacher generation. SAC should still use
    # the full train_cases list so saved teachers remain available for replay prefill.
    selected_cases = requested_cases
    skipped_cases: List[gps.GPSCase] = []
    if skip_existing_cases:
        print("SAC skip_existing_cases is ignored for actor/critic training; using all train_cases. Use the CEM stage skip flag to avoid recomputing teachers.")

    if selected_cases and sac_cfg.use_teacher_prefill:
        if use_saved_teacher_summary:
            teacher_output = prefill_from_saved_teachers(agent, selected_cases, traj_cfg, obj_cfg, sac_cfg, baseline_cache, save_dir, seed)
        if teacher_output is None:
            if allow_fresh_cem_teachers:
                teacher_output = prefill_with_gps_teachers(agent, selected_cases, traj_cfg, obj_cfg, sac_cfg, baseline_cache, teacher_cem_iters, teacher_population, seed, save_dir)
            else:
                print("No saved teachers found and fresh CEM teachers are disabled. Continuing without teacher prefill.")
    elif not selected_cases:
        print("No SAC training cases selected; saving metadata without interaction training.")

    run_record = dict(
        record_type="run",
        timestamp=str(np.datetime64("now")),
        resume_training=bool(resume_training),
        skip_existing_cases=bool(skip_existing_cases),
        total_interactions_requested=int(total_interactions),
        total_interactions_before=interactions_before,
        total_interactions=int(agent.total_interactions),
        teacher_cem_iters=int(teacher_cem_iters),
        teacher_population=int(teacher_population),
        eval_every=int(eval_every),
        save_every=int(save_every),
        model_path=checkpoint_path,
        trained_cases_this_run=[_case_to_dict(c) for c in selected_cases],
        skipped_cases=[_case_to_dict(c) for c in skipped_cases],
        replay_resumed=False,
        replay_note="Replay buffer is not checkpointed; resumed SAC runs reload saved CEM teachers and start with a refilled replay buffer.",
    )

    if selected_cases:
        for step in range(1, int(total_interactions) + 1):
            case = selected_cases[int(rng.integers(0, len(selected_cases)))]
            obs = case_obs(case)
            if agent.total_interactions < sac_cfg.start_random_steps:
                action = rng.uniform(-1.0, 1.0, size=traj_cfg.n_basis + 1).astype(np.float32)
            else:
                action = agent.act(obs, deterministic=False)
            ev = evaluate_action(case, action, traj_cfg, obj_cfg, sac_cfg, baseline_cache, seed=seed + 10000 + step)
            agent.replay.store(obs, action, ev["reward"], obs2=obs, done=True)
            agent.total_interactions += 1
            upd = agent.update(sac_cfg.updates_per_interaction)
            if step % max(1, eval_every) == 0 or step == 1:
                eval_rows = evaluate_sac_gps_policy(agent, selected_cases[:min(8, len(selected_cases))], traj_cfg, obj_cfg, sac_cfg, baseline_cache, seed=seed + 50000 + step, critic_refine=False)
                mean_existing = float(np.mean([r["existing_metrics"]["total_cost"] for r in eval_rows]))
                mean_actor = float(np.mean([r["actor_eval"]["cost"] for r in eval_rows]))
                mean_imp = 100.0 * (mean_existing - mean_actor) / max(abs(mean_existing), 1e-12)
                rec = dict(record_type="step", step=step, total_interactions=int(agent.total_interactions), train_reward=float(ev["reward"]), train_raw_reward=float(ev["raw_reward"]), train_cost=float(ev["cost"]), train_existing_cost=float(ev["existing_cost"]), mean_eval_existing=mean_existing, mean_eval_actor=mean_actor, mean_eval_improvement_pct=mean_imp, buffer=float(agent.replay.len), alpha=float(agent.alpha.numpy()))
                rec.update({k: float(v) for k, v in upd.items()})
                history.append(rec)
                print(f"SAC-GPS step {step:05d}/{total_interactions}: total_interactions={agent.total_interactions}, train_reward={ev['reward']:.4f}, actor_eval={mean_actor:.6g}, existing_eval={mean_existing:.6g}, improvement={mean_imp:.2f}%, buffer={agent.replay.len}")
            if save_dir and step % max(1, int(save_every)) == 0:
                run_record["total_interactions"] = int(agent.total_interactions)
                run_record["timestamp"] = str(np.datetime64("now"))
                _save_sac_artifacts(agent, save_dir, history, run_record, traj_cfg, obj_cfg, sac_cfg)
                print(f"SAC checkpoint saved at local step {step} (every {save_every} steps).")

    run_record["total_interactions"] = int(agent.total_interactions)
    run_record["timestamp"] = str(np.datetime64("now"))
    history.append(run_record)
    if save_dir:
        _save_sac_artifacts(agent, save_dir, history, run_record, traj_cfg, obj_cfg, sac_cfg)
    print(f"Total interactions after training: {agent.total_interactions}")
    return dict(agent=agent, history=history, teacher_output=teacher_output, baseline_cache=baseline_cache, agent_loaded=agent_loaded, train_cases=selected_cases, skipped_cases=skipped_cases)


#%% ========================= EVALUATION / PLOTS =========================

def evaluate_sac_gps_policy(agent, test_cases, traj_cfg, obj_cfg, sac_cfg, baseline_cache=None, seed=0, critic_refine=True):
    baseline_cache = baseline_cache or BaselineCache(obj_cfg)
    rows: List[Dict[str, object]] = []
    zero_action = np.zeros(traj_cfg.n_basis + 1, dtype=np.float32)
    for i, case in enumerate(test_cases):
        obs = case_obs(case)
        existing_metrics, existing_logs, existing_ref, existing_T = baseline_cache.get(case, seed=seed + i)
        actor_action = agent.act(obs, deterministic=True)
        actor_ev = evaluate_action(case, actor_action, traj_cfg, obj_cfg, sac_cfg, baseline_cache, seed=seed + 1000 + i)
        if critic_refine:
            refined_action = critic_refine_action(agent, obs, actor_action, sac_cfg.critic_refine_steps, sac_cfg.critic_refine_lr, sac_cfg.critic_refine_l2)
            refined_ev = evaluate_action(case, refined_action, traj_cfg, obj_cfg, sac_cfg, baseline_cache, seed=seed + 2000 + i)
        else:
            refined_action = actor_action.copy(); refined_ev = actor_ev
        zero_ev = evaluate_action(case, zero_action, traj_cfg, obj_cfg, sac_cfg, baseline_cache, seed=seed + 3000 + i)
        best_ev = min([actor_ev, refined_ev, zero_ev], key=lambda d: d["cost"])
        rows.append(dict(case=case, existing_metrics=existing_metrics, existing_logs=existing_logs, existing_ref=existing_ref, existing_T=existing_T, zero_eval=zero_ev, actor_eval=actor_ev, refined_eval=refined_ev, best_eval=best_ev, actor_metrics=actor_ev["metrics"], refined_metrics=refined_ev["metrics"], best_metrics=best_ev["metrics"]))
        print(f"eval {case.label()}: fixed-horizon LQR rollout objective={existing_metrics['total_cost']:.6g}, actor rollout objective={actor_ev['cost']:.6g}, refined rollout objective={refined_ev['cost']:.6g}, best={best_ev['cost']:.6g}, best_imp={best_ev['improvement_pct']:.2f}%")
    return rows


def _save_or_show(fig, save_dir, name, show):
    if save_dir:
        mkdir(save_dir); fig.savefig(os.path.join(save_dir, name), dpi=180, bbox_inches="tight")
    if show: plt.show()
    else: plt.close(fig)


def plot_training_history(history, save_dir=None, show=True):
    history = [row for row in history if row.get("record_type", "step") == "step" and "step" in row]
    if not history: return
    h = {k: np.array([row.get(k, np.nan) for row in history], dtype=float) for k in history[0].keys() if k != "record_type"}
    step = h["step"]
    fig = plt.figure(figsize=(8,5)); plt.plot(step, h["train_reward"], label="train reward"); plt.plot(step, h["train_raw_reward"], label="raw reward")
    plt.xlabel("interactions"); plt.ylabel("reward"); plt.title("SAC-GPS reward"); plt.grid(True, alpha=.4); plt.legend(); _save_or_show(fig, save_dir, "training_reward.png", show)


def plot_evaluation_summary(rows, obj_cfg, save_dir=None, show=True):
    if not rows: return
    labels = [f"{r['case'].theta_goal_deg:.0f}/{r['case'].alpha_deg:.0f}" for r in rows]
    x = np.arange(len(rows))
    existing = np.array([r["existing_metrics"]["total_cost"] for r in rows], float)
    actor = np.array([r["actor_eval"]["cost"] for r in rows], float)
    best = np.array([r["best_eval"]["cost"] for r in rows], float)
    fig = plt.figure(figsize=(max(9,.45*len(rows)),5)); plt.plot(x, existing, marker="o", label="fixed-horizon LQR rollout"); plt.plot(x, actor, marker="o", label="SAC actor"); plt.plot(x, best, marker="o", label="best")
    plt.xticks(x, labels, rotation=60, ha="right"); plt.ylabel("closed-loop rollout objective cost"); plt.title("Trajectory rollout objective comparison"); plt.grid(True, alpha=.4); plt.legend(); _save_or_show(fig, save_dir, "eval_cost_comparison.png", show)


def print_constraint_report(rows, obj_cfg):
    print("\n=== Constraint / improvement report ===")
    for r in rows:
        case = r["case"]; best = r["best_eval"]; m = best["metrics"]
        print(f"{case.label()}: fixed-horizon LQR rollout objective={best['existing_cost']:.6g}, best rollout objective={best['cost']:.6g}, imp={best['improvement_pct']:.2f}%, max|omega|={m['max_abs_omega']:.4g}/{obj_cfg.omega_limit}, max|tau_m|={m['max_abs_tau_m']:.4g}/{obj_cfg.torque_limit}, max|u|={m['max_abs_u_total']:.4g}/{obj_cfg.command_limit}, final_err={m['final_theta_error']:.4g}")


#%% ========================= USER SETTINGS =========================
TRAIN_GOAL_DEGS = (0, 30, 60, 90, 120, 150, 180)
TEST_GOAL_DEGS = (15, 45, 75, 105, 135, 165, 180)
TILT_DEGS = (0, 5, 10, 15, 20)
COUPLED_TILTS = True

SAVE_DIR = "true_gps_results_smoke"
SHOW_PLOTS = True
SEED = 0

# Stage switches
RUN_CEM_TEACHERS = True
RUN_SAC_TRAINING = True
RUN_EVALUATION = True

# Resume switches
RESUME_CEM_TEACHERS = True
RESUME_SAC_TRAINING = True
SKIP_EXISTING_CASES = True

# CEM teacher settings
TEACHER_CEM_ITERS = 4
TEACHER_POPULATION = 20
TEACHER_ELITE_FRAC = 0.25

# SAC settings
TOTAL_INTERACTIONS = 1000
EVAL_EVERY = 50
SAVE_EVERY = 50
USE_SAVED_TEACHER_SUMMARY = True
ALLOW_FRESH_CEM_TEACHERS = False


#%% ========================= BUILD CONFIGS AND CASES =========================
traj_cfg, obj_cfg, sac_cfg = make_default_configs()
mkdir(SAVE_DIR)
train_cases = gps.make_cases(TRAIN_GOAL_DEGS, TILT_DEGS, coupled_tilts=COUPLED_TILTS)
test_cases = gps.make_cases(TEST_GOAL_DEGS, TILT_DEGS, coupled_tilts=COUPLED_TILTS)

print("\n=== SAC-GPS two-stage resumable trajectory learning ===")
print(f"train goals={list(TRAIN_GOAL_DEGS)}, test goals={list(TEST_GOAL_DEGS)}")
print(f"tilts={list(TILT_DEGS)}, coupled={COUPLED_TILTS}")
print(f"train cases={len(train_cases)}, test cases={len(test_cases)}")
print("Plant/model/LQR/TDE+SMC rollout source: LQR_TrjOPt_TDESMCwithRLresidual.py and true_gps_lqr_guided_addon.py")
print(f"save_dir = {SAVE_DIR}")


#%% ========================= STAGE 1: RESUMABLE CEM TEACHERS =========================
teacher_stage_output = None
if RUN_CEM_TEACHERS:
    teacher_stage_output = generate_or_resume_cem_teachers(
        train_cases=train_cases,
        traj_cfg=traj_cfg,
        obj_cfg=obj_cfg,
        sac_cfg=sac_cfg,
        save_dir=SAVE_DIR,
        cem_iters=TEACHER_CEM_ITERS,
        population=TEACHER_POPULATION,
        elite_frac=TEACHER_ELITE_FRAC,
        seed=SEED,
        resume=RESUME_CEM_TEACHERS,
        skip_existing_cases=SKIP_EXISTING_CASES,
    )
else:
    print("Skipping CEM teacher stage because RUN_CEM_TEACHERS=False.")


#%% ========================= STAGE 2: RESUMABLE SAC TRAINING =========================
train_output = None
agent = None
baseline_cache = None
if RUN_SAC_TRAINING:
    train_output = train_sac_gps_agent(
        train_cases=train_cases,
        traj_cfg=traj_cfg,
        obj_cfg=obj_cfg,
        sac_cfg=sac_cfg,
        total_interactions=TOTAL_INTERACTIONS,
        teacher_cem_iters=TEACHER_CEM_ITERS,
        teacher_population=TEACHER_POPULATION,
        eval_every=EVAL_EVERY,
        seed=SEED,
        save_dir=SAVE_DIR,
        use_saved_teacher_summary=USE_SAVED_TEACHER_SUMMARY,
        allow_fresh_cem_teachers=ALLOW_FRESH_CEM_TEACHERS,
        resume_training=RESUME_SAC_TRAINING,
        skip_existing_cases=SKIP_EXISTING_CASES,
        save_every=SAVE_EVERY,
    )
    agent = train_output["agent"]
    baseline_cache = train_output["baseline_cache"]
else:
    print("Skipping SAC training stage because RUN_SAC_TRAINING=False.")


#%% ========================= STAGE 3: EVALUATION =========================
eval_rows = []
if RUN_EVALUATION:
    if agent is None:
        _print_tensorflow_diagnostics()
        agent = SACGPSAgent(obs_dim=3, act_dim=traj_cfg.n_basis + 1, cfg=sac_cfg)
        checkpoint_path = os.path.join(SAVE_DIR, "sac_gps_agent.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"RUN_EVALUATION=True but no SAC checkpoint exists at: {checkpoint_path}")
        agent.load(checkpoint_path)
        baseline_cache = BaselineCache(obj_cfg)
    eval_rows = evaluate_sac_gps_policy(agent, test_cases, traj_cfg, obj_cfg, sac_cfg, baseline_cache=baseline_cache, seed=SEED + 200000, critic_refine=True)
    print_constraint_report(eval_rows, obj_cfg)

    evaluation_table = []
    for r in eval_rows:
        case = r["case"]; best = r["best_eval"]; m = best["metrics"]
        evaluation_table.append(dict(theta_goal_deg=case.theta_goal_deg, alpha_deg=case.alpha_deg, phi_deg=case.phi_deg, existing_cost=best["existing_cost"], actor_cost=r["actor_eval"]["cost"], refined_cost=r["refined_eval"]["cost"], best_cost=best["cost"], improvement_pct=best["improvement_pct"], max_abs_omega=m["max_abs_omega"], max_abs_tau_m=m["max_abs_tau_m"], max_abs_u_total=m["max_abs_u_total"], final_theta_error=m["final_theta_error"], duration=m["duration"]))
    with open(os.path.join(SAVE_DIR, "evaluation_table.json"), "w") as f:
        json.dump(evaluation_table, f, indent=2)
    if train_output is not None:
        plot_training_history(train_output["history"], save_dir=SAVE_DIR, show=SHOW_PLOTS)
    plot_evaluation_summary(eval_rows, obj_cfg=obj_cfg, save_dir=SAVE_DIR, show=SHOW_PLOTS)
else:
    print("Skipping evaluation stage because RUN_EVALUATION=False.")

results = dict(agent=agent, train_output=train_output, teacher_stage_output=teacher_stage_output, eval_rows=eval_rows, train_cases=train_cases, test_cases=test_cases, traj_cfg=traj_cfg, obj_cfg=obj_cfg, sac_cfg=sac_cfg, save_dir=SAVE_DIR)
print("\nDone. Results are stored in variable: results")
print(f"Saved outputs to: {SAVE_DIR}")
