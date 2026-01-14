"""
aRLarm.py

Main entrypoint for training and playing the planar arm RL agent (SAC + attention).

Version 7.1:
    - Added ball physics target option with velocity verlet integration.
    - Added better naming identifiers for parallel training runs.
    - Activated Wandb logging callback during training.
    - Also log model config to metadata.
    - Do not set default values when calling keys to write to metadata.
"""

__version__ = "7.1"

import argparse
import math
import os
import json
from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn

import pygame
import zipfile

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime
import uuid

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
except ImportError:  # pragma: no cover - optional dependency
    wandb = None
    WandbCallback = None


# -----------------------------
# Kinematics + utility
# -----------------------------
def wrap_to_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi

def wrap_array_to_pi(arr: np.ndarray) -> np.ndarray:
    return (arr + np.pi) % (2.0 * np.pi) - np.pi

def fk_chain(thetas: Sequence[float], lengths: Sequence[float]):
    """Return list of (x, y, cumulative_theta) points for each linkage."""
    positions = []
    x = 0.0
    y = 0.0
    angle = 0.0
    for theta, length in zip(thetas, lengths):
        angle = wrap_to_pi(angle + theta)
        x += length * math.cos(angle)
        y += length * math.sin(angle)
        positions.append((x, y, angle))
    return positions

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# -----------------------------
# Ball physics
# -----------------------------
def get_acceleration(position, k=1.0, m=1.0):
    """
    Docstring for get_acceleration
    Calculates acceleration for a harmonic oscillator: F = -kx -> a = -(k/m)x
    
    :param position: Description
    :param k: Description
    :param m: Description

    :return: a: acceleration, float
    """
    return -(k/m) * position

def velocity_verlet_simulation(x0, v0, dt, t_max):
    """
    Docstring for velocity_verlet_simulation
    Runs a simulation using the velocity verlet integration method
    
    :param x0: (float) Initiate position
    :param v0: (float) Initial velocity
    :param dt: (float) Time step size
    :param t_max: (float) Total simulation time
    
    :return: times: np.ndarray of time steps, positions: np.ndarray of positions, velocities: np.ndarray of velocities
    """
    steps = max(int(t_max / max(dt, 1e-6)), 1) + 1
    times = np.linspace(0.0, dt * (steps - 1), steps)
    positions = np.zeros(steps, dtype=float)
    velocities = np.zeros(steps, dtype=float)
    positions[0] = x0
    velocities[0] = v0
    accel = get_acceleration(x0)
    for i in range(1, steps):
        x_new = positions[i - 1] + velocities[i - 1] * dt + 0.5 * accel * dt * dt
        accel_new = get_acceleration(x_new)
        v_new = velocities[i - 1] + 0.5 * (accel + accel_new) * dt
        positions[i] = x_new
        velocities[i] = v_new
        accel = accel_new
    return times, positions, velocities

# -----------------------------
# Environment
# -----------------------------
@dataclass
class ArmConfig:
    num_links: int = 2
    window_dim: Tuple[int, int] = (800, 800)
    reference_window: float = 800.0
    arm_scale: float = 2.1  # total arm length for reference window size
    target_radius_ratio: float = 0.06 / 2.1
    workspace_margin_ratio: float = 0.05 / 2.1
    render_padding_ratio: float = 0.2 / 2.1
    dt: float = 1.0
    max_dtheta: float = 0.15  # rad per step
    episode_len: int = 200
    reward_scale: float = 10.0
    action_penalty: float = 0.02
    success_bonus: float = 5.0
    jerk_penalty: float = 0.00
    jerk_threshold: float = 0.1

    version: str = "None"

    total_arm_length: float = field(init=False)
    link_lengths: Tuple[float, ...] = field(init=False)
    target_radius: float = field(init=False)
    workspace_margin: float = field(init=False)
    render_padding: float = field(init=False)

    def __post_init__(self):
        if self.num_links < 2:
            raise ValueError("num_links must be >= 2")
        min_dim = float(min(self.window_dim))
        scale = (min_dim / float(self.reference_window)) if self.reference_window else 1.0
        self.total_arm_length = self.arm_scale * scale
        per_link = self.total_arm_length / self.num_links
        self.link_lengths = tuple([per_link] * self.num_links)
        self.target_radius = self.target_radius_ratio * self.total_arm_length
        self.workspace_margin = self.workspace_margin_ratio * self.total_arm_length
        self.render_padding = self.render_padding_ratio * self.total_arm_length

@dataclass
class BallConfig:
    window_dim: Tuple[int, int] = (800, 800)
    reference_window: float = 800.0
    arm_scale: float = 2.1  # total arm length for reference window size
    total_arm_length: float = None
    ball_radius_ratio: float = 0.06 / 2.1
    workspace_margin_ratio: float = 0.05 / 2.1
    render_padding_ratio: float = 0.2 / 2.1
    dt: float = 0.01
    mass: float = 1.0
    damping: float = 0.995
    bounce: float = 0.5
    gravity: float = 0.025  # downward accel (world units / step^2)
    rest_speed_thresh: float = 0.1  # speed threshold to consider ball resting
    rest_pos_eps: float = 0.05  # y-distance to floor to consider resting

    ball_radius: float = field(init=False)
    workspace_margin: float = field(init=False)
    render_padding: float = field(init=False)

    def __post_init__(self):
        min_dim = float(min(self.window_dim))
        scale = (min_dim / float(self.reference_window)) if self.reference_window else 1.0
        total_length = self.total_arm_length if self.total_arm_length is not None else self.arm_scale * scale
        self.total_arm_length = total_length
        self.workspace_margin = self.workspace_margin_ratio * total_length
        self.ball_radius = self.ball_radius_ratio * total_length
        self.render_padding = self.render_padding_ratio * total_length


class PlanarArm(gym.Env):
    """
    Observation: (num_links+1 tokens, 4 features) = [per-joint tokens, target]
      features: [x, y, theta, type_id]
        - joint i: absolute position/orientation, type_id=i
        - target: workspace position + theta=0, type_id=num_links
    Action: joint deltas in [-1,1]^num_links scaled by max_dtheta.
    Reward: +scale*(prev_dist - new_dist) - action_penalty*||action||^2
            + success_bonus if within target_radius.
    """
    metadata = {"render_modes": ["human"], "render_fps": 25}

    def __init__(self, cfg: ArmConfig, render_mode=None, seed=0, use_ball_target: bool = False, ball_cfg: BallConfig = None):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode
        self.np_random = np.random.default_rng(seed)
        self.use_ball_target = bool(use_ball_target)
        self.ball_cfg = ball_cfg or BallConfig(
            window_dim=self.cfg.window_dim,
            reference_window=self.cfg.reference_window,
            arm_scale=self.cfg.arm_scale,
            total_arm_length=self.cfg.total_arm_length,
            dt=self.cfg.dt,
        )

        self.n_links = len(self.cfg.link_lengths)

        obs_shape = (self.n_links + 1, 4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_links,), dtype=np.float32
        )

        self.theta = np.zeros(self.n_links, dtype=np.float32)
        reach = self.cfg.total_arm_length - self.cfg.workspace_margin
        self.target = np.array([0.5 * reach, 0.0], dtype=np.float32)
        self.prev_dist = None
        self.t = 0
        self.prev_theta_vel = np.zeros(self.n_links, dtype=np.float32)
        self.prev_theta_accel = np.zeros(self.n_links, dtype=np.float32)
        self.jerk_history = []
        self.max_jerk_history = 240

        # pygame render state
        self._pg_inited = False
        self._screen = None
        self._clock = None
        self._W, self._H = self.cfg.window_dim

        # external target setter for mouse clicks in play mode
        self._pending_target = None

        # ball physics state (target replacement)
        self._ball_pos = np.zeros(2, dtype=np.float32)
        self._ball_vel = np.zeros(2, dtype=np.float32)
        self._ball_acc = np.zeros(2, dtype=np.float32)

    def _ball_workspace_limit(self) -> float:
        reach = self.cfg.total_arm_length - self.cfg.workspace_margin - self.ball_cfg.ball_radius
        return max(reach, 0.0)

    def _compute_ball_accel(self, position: np.ndarray) -> np.ndarray:
        """Constant gravity in -y."""
        return np.array([0.0, -abs(self.ball_cfg.gravity)], dtype=np.float32)

    def _set_ball_state(self, position: np.ndarray, velocity: np.ndarray = None):
        reach = self._ball_workspace_limit()
        clamped = np.array(
            [clamp(float(position[0]), -reach, reach), clamp(float(position[1]), -reach, reach)],
            dtype=np.float32,
        )
        self._ball_pos = clamped
        if velocity is not None:
            self._ball_vel = np.asarray(velocity, dtype=np.float32).reshape(2)
        else:
            self._ball_vel.fill(0.0)
        self._ball_acc = self._compute_ball_accel(self._ball_pos).astype(np.float32, copy=False)
        self.target = self._ball_pos.copy()

    def _init_ball_state(self):
        reach = self._ball_workspace_limit()
        if self._pending_target is not None:
            pos = self._pending_target.astype(np.float32)
            self._pending_target = None
        elif reach <= 0.0:
            pos = np.zeros(2, dtype=np.float32)
        else:
            r = reach * math.sqrt(self.np_random.random())
            a = 2 * math.pi * self.np_random.random()
            pos = np.array([r * math.cos(a), r * math.sin(a)], dtype=np.float32)

        speed = reach * 0.15 if reach > 0 else 0.0
        v_angle = 2 * math.pi * self.np_random.random()
        vel = np.array([math.cos(v_angle), math.sin(v_angle)], dtype=np.float32) * speed
        self._set_ball_state(pos, vel)

    def _apply_ball_bounds(self, pos: np.ndarray, vel: np.ndarray):
        reach = self._ball_workspace_limit()
        if reach <= 0.0:
            return np.zeros_like(pos)
        for axis in range(2):
            if pos[axis] > reach:
                pos[axis] = reach
                vel[axis] = -abs(vel[axis]) * self.ball_cfg.bounce
            elif pos[axis] < -reach:
                pos[axis] = -reach
                vel[axis] = abs(vel[axis]) * self.ball_cfg.bounce
        return pos

    def _update_ball(self):
        dt = max(self.ball_cfg.dt, 1e-6)
        accel = self._compute_ball_accel(self._ball_pos)
        new_pos = self._ball_pos + self._ball_vel * dt + 0.5 * accel * dt * dt
        new_accel = self._compute_ball_accel(new_pos)
        self._ball_vel = (self._ball_vel + 0.5 * (accel + new_accel) * dt) * self.ball_cfg.damping
        new_pos = self._apply_ball_bounds(new_pos, self._ball_vel)
        self._ball_acc = new_accel.astype(np.float32, copy=False)
        self._ball_pos = new_pos.astype(np.float32, copy=False)
        self.target = self._ball_pos.copy()

    def _ball_at_rest(self) -> bool:
        """Detect when ball is essentially stationary on the floor."""
        speed = float(np.linalg.norm(self._ball_vel))
        reach = self._ball_workspace_limit()
        on_floor = abs(self._ball_pos[1] + reach) < max(self.ball_cfg.rest_pos_eps, 1e-6)
        return on_floor and speed < max(self.ball_cfg.rest_speed_thresh, 1e-6)

    def set_target_world(self, x, y):
        # clamp to reachable annulus-ish workspace box (simple clamp)
        reach = max(self.cfg.total_arm_length - self.cfg.workspace_margin, 0.0)
        x = float(clamp(x, -reach, reach))
        y = float(clamp(y, -reach, reach))
        if self.use_ball_target:
            self._set_ball_state(np.array([x, y], dtype=np.float32))
            return
        self._pending_target = np.array([x, y], dtype=np.float32)

    def _sample_target(self):
        # sample uniformly in a disk of radius reach, reject too-close-to-origin if desired
        reach = max(self.cfg.total_arm_length - self.cfg.workspace_margin, 0.0)
        for _ in range(1000):
            r = reach * math.sqrt(self.np_random.random())
            a = 2 * math.pi * self.np_random.random()
            x = r * math.cos(a)
            y = r * math.sin(a)
            return np.array([x, y], dtype=np.float32)
        return np.array([0.5 * reach, 0.0], dtype=np.float32)

    def _get_obs(self):
        positions = fk_chain(self.theta, self.cfg.link_lengths)
        tokens = []
        for idx, (x, y, theta_abs) in enumerate(positions):
            tokens.append([x, y, theta_abs, float(idx)])
        tokens.append([self.target[0], self.target[1], 0.0, float(self.n_links)])
        return np.asarray(tokens, dtype=np.float32)

    def _dist_to_target(self):
        positions = fk_chain(self.theta, self.cfg.link_lengths)
        x_end, y_end, _ = positions[-1]
        dx = x_end - float(self.target[0])
        dy = y_end - float(self.target[1])
        return math.sqrt(dx * dx + dy * dy)

    def reset(self, seed=None, options=None):
        if options is not None and options.get("soft"):
            return self.soft_reset()
        super().reset(seed=seed)
        self.t = 0
        self.theta = self.np_random.uniform(-math.pi, math.pi, size=self.n_links).astype(np.float32)
        self.prev_theta_vel.fill(0.0)
        self.prev_theta_accel.fill(0.0)
        self.jerk_history.clear()

        if self.use_ball_target:
            self._init_ball_state()
        else:
            if self._pending_target is not None:
                self.target = self._pending_target.copy()
                self._pending_target = None
            elif self.render_mode != "human":
                self.target = self._sample_target()

        self.prev_dist = self._dist_to_target()
        obs = self._get_obs()
        info = {"dist": self.prev_dist}
        return obs, info

    def soft_reset(self):
        # Reset episode state without changing joint angles.
        self.t = 0
        self.prev_theta_vel.fill(0.0)
        self.prev_theta_accel.fill(0.0)

        if self.use_ball_target:
            self._init_ball_state()
        elif self._pending_target is not None:
            self.target = self._pending_target.copy()
            self._pending_target = None

        self.prev_dist = self._dist_to_target()
        obs = self._get_obs()
        info = {"dist": self.prev_dist}
        return obs, info

    def step(self, action):
        self.t += 1
        action = np.asarray(action, dtype=np.float32).reshape(self.n_links)
        action = np.clip(action, -1.0, 1.0)

        prev_theta = self.theta.copy()
        dtheta = self.cfg.max_dtheta * action
        self.theta = wrap_array_to_pi(self.theta + dtheta * self.cfg.dt)
        dt = max(self.cfg.dt, 1e-6)
        # Use wrapped delta to avoid velocity spikes at the +/-pi boundary.
        theta_delta = wrap_array_to_pi(self.theta - prev_theta)
        theta_vel = theta_delta / dt
        theta_accel = (theta_vel - self.prev_theta_vel) / dt
        theta_jerk = (theta_accel - self.prev_theta_accel) / dt
        jerk_mag = float(np.linalg.norm(theta_jerk))
        self.prev_theta_vel = theta_vel.astype(np.float32, copy=False)
        self.prev_theta_accel = theta_accel.astype(np.float32, copy=False)
        self._record_jerk(jerk_mag)

        # apply pending mouse target immediately (for interactive play)
        ball_rest = False
        if self.use_ball_target:
            if self._pending_target is not None:
                self._set_ball_state(self._pending_target)
                self._pending_target = None
            self._update_ball()
            ball_rest = self._ball_at_rest()
        elif self._pending_target is not None:
            self.target = self._pending_target.copy()
            self._pending_target = None

        dist = self._dist_to_target()
        # reward: positive if closer, negative if farther (exact sign from delta)
        delta = self.prev_dist - dist
        reward = self.cfg.reward_scale * delta
        reward -= self.cfg.action_penalty * float(np.dot(action, action))
        if jerk_mag > self.cfg.jerk_threshold:
            reward -= self.cfg.jerk_penalty * jerk_mag

        success = dist <= self.cfg.target_radius * 1.0
        terminated = success
        if self.use_ball_target and ball_rest:
            terminated = True
        if terminated:
            reward += self.cfg.success_bonus

        truncated = False if self.use_ball_target else self.t >= self.cfg.episode_len
        self.prev_dist = dist

        obs = self._get_obs()
        info = {"dist": dist, "jerk": jerk_mag, "ball_rest": ball_rest, "success": success}
        return obs, reward, terminated, truncated, info

    def _record_jerk(self, jerk_value: float):
        self.jerk_history.append(jerk_value)
        if len(self.jerk_history) > self.max_jerk_history:
            self.jerk_history = self.jerk_history[-self.max_jerk_history:]

    # -----------------------------
    # Rendering (pygame)
    # -----------------------------
    def _world_to_screen(self, x, y):
        # world coords roughly in [-reach, reach]
        reach = self.cfg.total_arm_length + self.cfg.render_padding
        sx = int((x / (2 * reach) + 0.5) * self._W)
        sy = int((0.5 - y / (2 * reach)) * self._H)
        return sx, sy

    def _screen_to_world(self, sx, sy):
        reach = self.cfg.total_arm_length + self.cfg.render_padding
        x = ((sx / self._W) - 0.5) * (2 * reach)
        y = (0.5 - (sy / self._H)) * (2 * reach)
        return x, y

    def render(self, metadata=None, target_mode="m", target_xy=None, target_path=None, hide_info=False):
        if self.render_mode != "human":
            return

        if not self._pg_inited:
            pygame.init()
            pygame.display.set_caption("4-DOF Planar Arm (click to set target)")
            self._screen = pygame.display.set_mode((self._W, self._H))
            self._clock = pygame.time.Clock()
            self._pg_inited = True

        # handle events (mouse click -> target)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.MOUSEBUTTONDOWN and target_mode == "m":
                sx, sy = pygame.mouse.get_pos()
                x, y = self._screen_to_world(sx, sy)
                self.set_target_world(x, y)
            elif target_xy is not None and target_mode == "p":
                x, y = target_xy
                self.set_target_world(x, y)
            elif event.type == pygame.MOUSEBUTTONDOWN and target_mode == "b":
                sx, sy = pygame.mouse.get_pos()
                x, y = self._screen_to_world(sx, sy)
                self.set_target_world(x, y)
            else:
                pass  # ignore other events

        self._screen.fill((18, 18, 18))

        positions = fk_chain(self.theta, self.cfg.link_lengths)
        base = (0.0, 0.0)

        joint_points = [base] + [(x, y) for (x, y, _) in positions]
        screen_points = [self._world_to_screen(*pt) for pt in joint_points]
        tgt = self._world_to_screen(float(self.target[0]), float(self.target[1]))

        # draw arm
        for start, end in zip(screen_points[:-1], screen_points[1:]):
            pygame.draw.line(self._screen, (220, 220, 220), start, end, 6)
        for idx, pt in enumerate(screen_points):
            color = (255, 200, 120) if idx == len(screen_points) - 1 else (120, 200, 255)
            pygame.draw.circle(self._screen, color, pt, 8)

        # draw target
        pygame.draw.circle(self._screen, (120, 255, 140), tgt, 10, 2)

        # draw target path if provided
        for path_pt in (target_path or []):
            sp = self._world_to_screen(path_pt[0], path_pt[1])
            pygame.draw.circle(self._screen, (255, 120, 120), sp, 4)

        ### INFORMATION OVERLAYS
        if not hide_info:
            # draw jerk scope
            self._draw_jerk_scope()

        # display metadata
        if metadata is not None and not hide_info:
            font = pygame.font.SysFont("Arial", 16)
            lines = []
            if "steps" in metadata:
                lines.append(f"Trained steps: {metadata['steps']}")
            if "jerk_penalty" in metadata:
                lines.append(f"Jerk penalty: {metadata['jerk_penalty']:.4f}")
            if "jerk_threshold" in metadata:
                lines.append(f"Jerk threshold: {metadata['jerk_threshold']:.4f}")
            if "notes" in metadata:
                lines.append(f"Notes: {metadata['notes']}")
            for i, line in enumerate(lines):
                text_surf = font.render(line, True, (200, 200, 200))
                self._screen.blit(text_surf, (10, 10 + i * 20))

        # fps cap
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def capture_frame(self):
        if not self._pg_inited or self._screen is None:
            return None
        frame = pygame.surfarray.array3d(self._screen)
        return np.transpose(frame, (1, 0, 2))

    def _draw_jerk_scope(self):
        rect_width = 260
        rect_height = 120
        margin = 15
        rect_x = self._W - rect_width - margin
        rect_y = self._H - rect_height - margin
        rect = pygame.Rect(rect_x, rect_y, rect_width, rect_height)
        pygame.draw.rect(self._screen, (15, 15, 25), rect)
        pygame.draw.rect(self._screen, (60, 60, 80), rect, 1)

        if len(self.jerk_history) < 2:
            return

        hist = self.jerk_history[-min(self.max_jerk_history, 100) :]
        if not hasattr(self, "_max_jerk_val"):
            self._max_jerk_val = 1e-6
        max_val = max(max(hist), self._max_jerk_val)
        self._max_jerk_val = max_val
        usable_height = rect_height - 10
        usable_width = rect_width - 10
        step_x = usable_width / (len(hist) - 1) if len(hist) > 1 else usable_width
        points = []
        for idx, value in enumerate(hist):
            x = rect.x + 5 + idx * step_x
            y = rect.bottom - 5 - (value / max_val) * usable_height
            points.append((x, y))
        if len(points) >= 2:
            pygame.draw.lines(self._screen, (120, 255, 120), False, points, 2)

        font = pygame.font.SysFont("Arial", 14)
        label = font.render("Jerk magnitude", True, (200, 200, 200))
        self._screen.blit(label, (rect.x, rect.y - 18))
        max_text = font.render(f"max {max_val:.3f}", True, (200, 200, 200))
        self._screen.blit(max_text, (rect.x, rect.y + rect_height + 2))

    def close(self):
        if self._pg_inited:
            pygame.quit()
            self._pg_inited = False


# -----------------------------
# Attention feature extractor (Transformer over tokens)
# -----------------------------
class TokenTransformerExtractor(BaseFeaturesExtractor):
    """
    Expects obs shape: (n_tokens, feat_dim) per env.
    For SB3, obs arrives as tensor (batch, n_tokens, feat_dim).
    Outputs pooled embedding (batch, d_model).
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__(observation_space, features_dim=d_model)
        n_tokens, feat_dim = observation_space.shape
        self.n_tokens = n_tokens

        self.in_proj = nn.Linear(feat_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # learned token-type embedding sized to number of tokens (joint count + target)
        self.type_emb = nn.Embedding(n_tokens, d_model)

        # pool via attention over tokens (learned query)
        self.pool_q = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pool_q, std=0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, T, F)
        x = obs

        # last feature is type_id (float), convert to long indices
        type_id = torch.clamp(x[..., 3].round().long(), 0, self.n_tokens - 1)
        x = self.in_proj(x) + self.type_emb(type_id)

        x = self.enc(x)
        x = self.norm(x)

        B = x.shape[0]
        q = self.pool_q.expand(B, -1, -1)  # (B,1,D)
        pooled, _ = self.pool_attn(q, x, x, need_weights=False)  # (B,1,D)
        return pooled[:, 0, :]  # (B,D)


# -----------------------------
# Training diagnostics
# -----------------------------
class WandbDiagnosticsCallback(BaseCallback):
    def __init__(self, log_freq: int = 1000):
        super().__init__()
        self.log_freq = max(int(log_freq), 1)
        self._last_log = 0

    def _on_step(self) -> bool:
        if wandb is None or wandb.run is None:
            return True
        if self.num_timesteps - self._last_log < self.log_freq:
            return True
        self._last_log = self.num_timesteps

        data = {"time/num_timesteps": self.num_timesteps}
        infos = self.locals.get("infos") or []
        rewards = self.locals.get("rewards")

        if rewards is not None:
            data["train/reward_mean"] = float(np.mean(rewards))

        if infos:
            dists = [info.get("dist") for info in infos if "dist" in info]
            if dists:
                data["train/dist_mean"] = float(np.mean(dists))
            jerks = [info.get("jerk") for info in infos if "jerk" in info]
            if jerks:
                data["train/jerk_mean"] = float(np.mean(jerks))
            successes = [info.get("success") for info in infos if "success" in info]
            if successes:
                data["train/success_rate"] = float(np.mean(successes))
            ep_infos = [info.get("episode") for info in infos if "episode" in info]
            if ep_infos:
                ep_rew = [ep["r"] for ep in ep_infos if "r" in ep]
                ep_len = [ep["l"] for ep in ep_infos if "l" in ep]
                if ep_rew:
                    data["rollout/ep_rew_mean"] = float(np.mean(ep_rew))
                if ep_len:
                    data["rollout/ep_len_mean"] = float(np.mean(ep_len))

        logger = getattr(self.model, "logger", None)
        name_to_value = getattr(logger, "name_to_value", None)
        if isinstance(name_to_value, dict):
            for key in (
                "train/actor_loss",
                "train/critic_loss",
                "train/ent_coef",
                "train/ent_coef_loss",
                "train/qf1_loss",
                "train/qf2_loss",
                "train/qf1_mean",
                "train/qf2_mean",
            ):
                if key in name_to_value:
                    data[key] = float(name_to_value[key])

        wandb.log(data, step=self.num_timesteps)
        return True


# -----------------------------
# Train / Play / Trace
# -----------------------------
def make_env(render_mode=None, seed=0, use_ball_target: bool = False, ball_cfg: BallConfig = None, **cfg_kwargs):
    cfg = ArmConfig(**cfg_kwargs)
    return PlanarArm(cfg, render_mode=render_mode, seed=seed, use_ball_target=use_ball_target, ball_cfg=ball_cfg)


def train(
    model_path: str,
    model_cfg: dict,
    steps: int,
    seed: int,
    wandb_cfg=None,
    use_ball_target: bool = False,
    ball_cfg: BallConfig = None,
    **cfg_kwargs
):
    """
    Trains a Soft Actor-Critic (SAC) model on a custom environment, optionally integrating with Weights & Biases (W&B) for experiment tracking.

    Args:
        model_path (str): Path to save the trained model.
        model_cfg (dict): Configuration dictionary for the model architecture, including transformer parameters.
        steps (int): Number of training timesteps.
        seed (int): Random seed for reproducibility.
        wandb_cfg (dict, optional): Configuration for W&B logging. If None or not enabled, W&B is not used.
        use_ball_target (bool, optional): Whether to use a ball target in the environment. Defaults to False.
        ball_cfg (BallConfig, optional): Configuration for the ball target, if used.
        **cfg_kwargs: Additional keyword arguments for environment configuration.

    Details:
        - Initializes a vectorized environment using the provided configuration.
        - Defines a custom policy with a Transformer-based feature extractor.
        - Creates and trains a SAC model with the specified architecture and hyperparameters.
        - Optionally logs training metrics and saves models using W&B if enabled.
        - Saves training metadata (including W&B run info if applicable) into the model file.

    Notes:
        - The `dim_ff` parameter in `model_cfg` refers to the dimensionality of the feedforward (FF) network within the Transformer feature extractor. It controls the size of the hidden layer in the transformer's feedforward subnetwork, typically set larger than `d_model` for increased model capacity.
    """
    def _thunk():
        env = make_env(
            render_mode=None,
            seed=seed,
            use_ball_target=use_ball_target,
            ball_cfg=ball_cfg,
            **cfg_kwargs,
        )
        return Monitor(env)

    # Create vectorized environment
    vec_env = DummyVecEnv([_thunk])

    d_model = model_cfg.get("d_model", 128)
    nhead = model_cfg.get("nhead", 4)
    num_layers = model_cfg.get("num_layers", 2)
    dim_ff = model_cfg.get("dim_ff", 256)
    dropout = model_cfg.get("dropout", 0.0)
    # Define custom policy with Transformer feature extractor
    policy_kwargs = dict(
        features_extractor_class=TokenTransformerExtractor,
        features_extractor_kwargs=dict(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_ff=dim_ff,
            dropout=dropout,
        ),
        net_arch=dict(
            pi=[256, 256],
            qf=[256, 256],
        ),
    )

    # Create the SAC model
    model = SAC(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.98,
        tau=0.02,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        buffer_size=200_000,
    )

    # Weights & Biases integration
    wandb_run = None
    wandb_callback = None
    diag_callback = None
    wandb_enabled = bool(wandb_cfg and wandb_cfg.get("enable"))
    if wandb_enabled:
        if wandb is None or WandbCallback is None:
            raise RuntimeError(
                "Weights & Biases logging requested but the wandb package is not installed."
            )
        # Strip the model path for the base name and use that as the wandb run_name
        model_base_name = os.path.splitext(os.path.basename(model_path))[0]
        run_name = wandb_cfg.get("run_name") or model_base_name
        wandb_project = wandb_cfg.get("project") or "aRLarm"
        wandb_entity = wandb_cfg.get("entity")
        wandb_config = dict(
            seed=seed,
            steps=steps,
            model_path=model_path,
            **cfg_kwargs,
        )
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config=wandb_config,
            save_code=True,
        )
        save_root = os.path.join("wandb_models", wandb_run.id or run_name)
        os.makedirs(save_root, exist_ok=True)
        wandb_callback = WandbCallback(
            gradient_save_freq=0,
            model_save_path=save_root,
            model_save_freq=0,
            verbose=1,
        )
        diag_callback = WandbDiagnosticsCallback(log_freq=1000)

    # Train the model
    callbacks = None
    if wandb_callback is not None and diag_callback is not None:
        callbacks = [wandb_callback, diag_callback]
    elif wandb_callback is not None:
        callbacks = wandb_callback
    elif diag_callback is not None:
        callbacks = diag_callback

    model.learn(total_timesteps=steps, progress_bar=True, callback=callbacks)
    model.save(model_path)

    # Save metadata
    meta = {
        "steps": steps,
        "num_links": cfg_kwargs["num_links"],
        "jerk_penalty": cfg_kwargs["jerk_penalty"],
        "jerk_threshold": cfg_kwargs["jerk_threshold"],
        "version": cfg_kwargs["version"],
        "model_cfg": model_cfg,
        "window_dim": cfg_kwargs["window_dim"],
        "arm_scale": cfg_kwargs["arm_scale"],
        "seed": seed,
        "notes": "Iterative study of whether jerk penalty improves performance.",
    }
    # Add W&B info to metadata if available
    if wandb_run is not None:
        meta["wandb_run"] = wandb_run.name or wandb_run.id
        meta["wandb_project"] = wandb_cfg.get("project") or "aRLarm"

    # Append metadata to the model zip file
    with zipfile.ZipFile(model_path, "a") as zf:
        zf.writestr("meta.json", json.dumps(meta))

    # Cleanup
    vec_env.close()
    if wandb_run is not None:
        wandb.save(os.path.abspath(model_path))
        wandb_run.finish()

def read_metadata(model_path: str):
    with zipfile.ZipFile(model_path, "r") as zf:
        if "meta.json" in zf.namelist():
            meta_str = json.loads(zf.read("meta.json").decode("utf-8"))
            return meta_str
    return None

def play(model_path: str, seed: int, use_ball_target: bool = False, ball_cfg: BallConfig = None, hide_info: bool = False, **cfg_kwargs):
    env = make_env(render_mode="human", seed=seed, use_ball_target=use_ball_target, ball_cfg=ball_cfg, **cfg_kwargs)
    model = SAC.load(model_path)
    meta = read_metadata(model_path)

    obs, info = env.reset()
    target_mode = "b" if use_ball_target else "m"
    while True:
        env.render(metadata=meta, target_mode=target_mode, hide_info=hide_info)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

def trace(
    model_path: str,
    seed: int,
    traj: Sequence[Tuple[float, float]],
    record_video: bool = False,
    max_frames: int = 1000,
    output_dir: str = ".",
    hide_info: bool = False,
    **cfg_kwargs,
    ):
    """
    Trace a predefined trajectory of target positions.
    Records a video if requested.
    """
    env = make_env(render_mode="human", seed=seed, use_ball_target=False, ball_cfg=None, **cfg_kwargs)
    model = SAC.load(model_path)
    meta = read_metadata(model_path)

    obs, info = env.reset()
    traj_idx = 0
    traj_len = len(traj)
    num_frames = 0
    completed_targets = 0
    video_writer = None
    if record_video and traj_len > 0:
        try:
            import imageio.v2 as imageio
        except ImportError:
            print("imageio not installed; skipping video recording.")
            record_video = False
        else:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            video_dir = os.path.join(output_dir, "videos", model_name)
            os.makedirs(video_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(video_dir, f"trace_{timestamp}.mp4")
            video_writer = imageio.get_writer(
                video_path,
                fps=env.metadata["render_fps"],
                codec="libx264",
                quality=8,
            )
    while True:
        target_xy = traj[traj_idx % traj_len]
        env.render(metadata=meta, target_mode="p", target_xy=target_xy, target_path=traj, hide_info=hide_info)
        if record_video and video_writer is not None:
            frame = env.capture_frame()
            if frame is not None:
                video_writer.append_data(frame)
                num_frames += 1
                if max_frames > 0 and num_frames >= max_frames:
                    break
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            traj_idx += 1
            completed_targets += 1
            next_target_xy = traj[traj_idx % traj_len]
            env.set_target_world(next_target_xy[0], next_target_xy[1])
            obs, info = env.reset(options={"soft": True})
            if record_video and completed_targets >= traj_len:
                break
    if video_writer is not None:
        video_writer.close()
        print(f"Video saved to: {video_path}")

### -----------------------------
# Trajectory generators for path-following mode
### -----------------------------
def generate_wavy_traj(num_points: int, reach: float, amplitude: float, frequency: float, center=(0.25, 0.0)):
    traj = []
    for i in range(num_points):
        t = i / (num_points - 1)
        x = center[0] + (t - 0.5) * 2 * reach * 0.8  # span 80% of reach
        y = center[1] + amplitude * math.sin(2 * math.pi * frequency * t)
        traj.append((x, y))
    return traj

def generate_star_traj(num_points: int, reach: float, spikes: int = 5, center=(0.15, 0.15)):
    traj = []
    if num_points <= 0:
        return traj

    inner_radius = reach * 0.5
    total_vertices = spikes * 2
    angle_step = math.pi / spikes
    start_angle = -math.pi / 2.0
    vertices = []
    for i in range(total_vertices):
        angle = start_angle + i * angle_step
        radius = reach if i % 2 == 0 else inner_radius
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        vertices.append((x, y))

    for i in range(num_points):
        t = (i / num_points) * total_vertices
        edge_idx = int(t) % total_vertices
        local_t = t - edge_idx
        x0, y0 = vertices[edge_idx]
        x1, y1 = vertices[(edge_idx + 1) % total_vertices]
        x = x0 + (x1 - x0) * local_t
        y = y0 + (y1 - y0) * local_t
        traj.append((x, y))
    return traj

def generate_circle_traj(num_points: int, reach: float, center=(0.15, 0.15)):
    traj = []
    for i in range(num_points):
        angle = (2 * math.pi * i) / num_points
        x = center[0] + reach * math.cos(angle)
        y = center[1] + reach * math.sin(angle)
        traj.append((x, y))
    return traj

### -----------------------------
# Runner / Argument Parsing
### -----------------------------
def main(argv=None):

    VERSION = __version__
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_model_name_root = f"rlarm_v{VERSION}"
    
    ap = argparse.ArgumentParser()
    # Core operation modes
    ap.add_argument("--train", action="store_true", help="Train a new model")
    ap.add_argument("--play", action="store_true", help="Play using a trained model")
    ap.add_argument("--model", type=str, default=f"{default_model_name_root}.zip", help="Model filename")
    ap.add_argument("--output-dir", type=str, default="./models/", help="Directory to save/load models")
    ap.add_argument("--steps", type=int, default=100, help="Number of training steps")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")

    # Model/Transformer architecture
    ap.add_argument("--model-d-model", type=int, default=128, help="Transformer model dimension")
    ap.add_argument("--model-nhead", type=int, default=4, help="Number of attention heads in the Transformer")
    ap.add_argument("--model-num-layers", type=int, default=2, help="Number of Transformer layers")
    ap.add_argument("--model-dim-ff", type=int, default=256, help="Feedforward network dimension in the Transformer")
    ap.add_argument("--model-dropout", type=float, default=0.0, help="Dropout rate in the Transformer")

    # Target/trajectory options
    ap.add_argument("--target", type=str, choices=["m", "p", "b"], default="m",
                    help="Target mode: 'm' for mouse, 'p' for path, 'b' for ball")
    ap.add_argument("--target-shape", type=str, choices=["wavy", "star", "circle"], default="wavy",
                    help="Predefined target path shape for 'p' mode")
    ap.add_argument("--record-video", action="store_true", help="Record video during play mode")
    ap.add_argument("--max-frames", type=int, default=250, help="Cap the number of frames so that all videos are finite")
    ap.add_argument("--clean", action="store_true", help="Hide info overlays for output video")

    # Arm/environment configuration
    ap.add_argument("--num-links", type=int, default=2, help="Number of arm links")
    ap.add_argument("--window-width", type=int, default=800, help="Window width for rendering/scaling")
    ap.add_argument("--window-height", type=int, default=800, help="Window height for rendering/scaling")
    ap.add_argument("--arm-scale", type=float, default=2.1,
                    help="Reference total arm length for reference-window size")
    ap.add_argument("--reference-window", type=float, default=800.0,
                    help="Reference window size for scaling arm lengths")

    # Reward/penalty configuration
    ap.add_argument("--jerk-penalty", type=float, default=0.0, help="Penalty weight for jerk in reward")
    ap.add_argument("--jerk-threshold", type=float, default=0.1, help="Jerk threshold for penalty")

    # Weights & Biases logging
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    ap.add_argument("--wandb-project", type=str, default=None, help="Weights & Biases project name")
    ap.add_argument("--wandb-entity", type=str, default=None, help="Weights & Biases entity/user name")
    ap.add_argument("--wandb-run-name", type=str, default=None, help="Custom run name for Weights & Biases")

    args = ap.parse_args(argv)
    if args.model == f"{default_model_name_root}.zip":
        run_suffix = uuid.uuid4().hex[:8]
        args.model = f"{default_model_name_root}[{args.num_links}]_{args.steps}_{timestamp}_{run_suffix}.zip"

    # Training configurations
    # This is passed to the arm configuration
    cfg_kwargs = dict(
        num_links=args.num_links,
        window_dim=(args.window_width, args.window_height),
        arm_scale=args.arm_scale,
        reference_window=args.reference_window,
        jerk_penalty=args.jerk_penalty,
        jerk_threshold=args.jerk_threshold,
        version=VERSION,
    )
    # This is passed to the model configuration
    model_cfg = dict(
        d_model=args.model_d_model,
        nhead=args.model_nhead,
        num_layers=args.model_num_layers,
        dim_ff=args.model_dim_ff,
        dropout=args.model_dropout,
    )
    model_path = os.path.join(args.output_dir, args.model)
    # Weights and biases config
    wandb_kwargs = dict(
        enable=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
    )
    # Play settings
    use_ball_target = args.target == "b"
    ball_cfg = None  # placeholder for future custom ball configs
    
    # Call training pipeline
    if args.train:
        train(
            model_path,
            model_cfg,
            args.steps,
            args.seed,
            wandb_cfg=wandb_kwargs,
            use_ball_target=use_ball_target,
            ball_cfg=ball_cfg,
            **cfg_kwargs,
        )
    # Call play or trace pipeline
    if args.play and args.target == "p":
        # Generate trajectory based on selected shape
        reach = cfg_kwargs["arm_scale"] * (min(args.window_width, args.window_height) / args.reference_window)
        if args.target_shape == "wavy":
            traj = generate_wavy_traj(num_points=200, reach=reach * 0.8, amplitude=reach * 0.3, frequency=2)
        elif args.target_shape == "star":
            traj = generate_star_traj(num_points=200, reach=reach * 0.8, spikes=5)
        elif args.target_shape == "circle":
            traj = generate_circle_traj(num_points=200, reach=reach * 0.8)
        else:
            raise ValueError(f"Unknown or missing target shape: {args.target_shape}")
        trace(
            model_path,
            args.seed,
            traj,
            record_video=args.record_video,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            hide_info=args.clean,
            **cfg_kwargs,
        )
        return
    if args.play:
        play(model_path, args.seed, use_ball_target=use_ball_target, ball_cfg=ball_cfg, hide_info=args.clean, **cfg_kwargs)
        return
    if not args.train and not args.play:
        print("Use --train and/or --play")

if __name__ == "__main__":
    main()
