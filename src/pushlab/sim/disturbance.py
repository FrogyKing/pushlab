"""Stochastic external disturbance model d_t ~ D for both NumPy and JAX/MJX backends."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from pushlab.config import DisturbanceConfig


@dataclass
class WrenchEvent:
    """Represents a 6-DOF external wrench applied to the robot trunk."""

    force_xyz: np.ndarray  # shape (3,) in Newtons [Fx, Fy, Fz]
    torque_xyz: np.ndarray  # shape (3,) in N*m [Tx, Ty, Tz]
    start_step: int
    end_step: int
    magnitude_n: float
    azimuth_rad: float

    @property
    def wrench_6d(self) -> np.ndarray:
        return np.concatenate([self.force_xyz, self.torque_xyz]).astype(np.float64)


class DisturbanceSchedule:
    """Deterministic-seed disturbance generator for single-environment evaluation and interactive pushes."""

    def __init__(self, config: DisturbanceConfig, seed: int = 42) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.active_event: WrenchEvent | None = None
        self.event_history: list[WrenchEvent] = []
        self._next_push_step: int = self._sample_next_interval(0)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.active_event = None
        self.event_history.clear()
        self._next_push_step = self._sample_next_interval(0)

    def _sample_next_interval(self, current_step: int) -> int:
        base = max(10, self.config.push_interval_steps)
        jitter = int(self.rng.integers(-base // 5, base // 5 + 1))
        return current_step + max(15, base + jitter)

    def trigger_manual_push(
        self,
        current_step: int,
        force_magnitude_n: float,
        azimuth_rad: float,
        duration_steps: int = 8,
        vertical_force_n: float = 0.0,
        torque_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> WrenchEvent:
        """Apply an explicit interactive push wrench starting at current_step."""
        fx = float(force_magnitude_n * np.cos(azimuth_rad))
        fy = float(force_magnitude_n * np.sin(azimuth_rad))
        fz = float(vertical_force_n)
        event = WrenchEvent(
            force_xyz=np.array([fx, fy, fz], dtype=np.float64),
            torque_xyz=np.array(torque_xyz, dtype=np.float64),
            start_step=current_step,
            end_step=current_step + max(1, duration_steps),
            magnitude_n=float(np.hypot(fx, fy)),
            azimuth_rad=float(azimuth_rad),
        )
        self.active_event = event
        self.event_history.append(event)
        return event

    def sample_random_event(self, current_step: int) -> WrenchEvent:
        """Sample a stochastic wrench d_t ~ D according to DisturbanceConfig."""
        f_min, f_max = self.config.force_magnitude_range
        mag = float(self.rng.uniform(f_min, f_max)) if f_max > 0 else 0.0
        a_min, a_max = self.config.angle_range_rad
        azimuth = float(self.rng.uniform(a_min, a_max))
        fx = mag * np.cos(azimuth)
        fy = mag * np.sin(azimuth)
        max_fz = mag * self.config.vertical_force_fraction
        fz = float(self.rng.uniform(-max_fz, max_fz)) if max_fz > 0 else 0.0

        t_min, t_max = self.config.torque_magnitude_range
        if t_max > 0:
            t_mag = float(self.rng.uniform(t_min, t_max))
            t_dir = self.rng.normal(size=3)
            t_dir /= np.linalg.norm(t_dir) + 1e-8
            torque = t_mag * t_dir
        else:
            torque = np.zeros(3, dtype=np.float64)

        event = WrenchEvent(
            force_xyz=np.array([fx, fy, fz], dtype=np.float64),
            torque_xyz=torque.astype(np.float64),
            start_step=current_step,
            end_step=current_step + max(1, self.config.push_duration_steps),
            magnitude_n=mag,
            azimuth_rad=azimuth,
        )
        return event

    def get_wrench_at_step(self, step: int) -> np.ndarray:
        """Return the 6D wrench [Fx, Fy, Fz, Tx, Ty, Tz] active at the given control step."""
        if self.active_event is not None:
            if self.active_event.start_step <= step < self.active_event.end_step:
                return self.active_event.wrench_6d
            if step >= self.active_event.end_step:
                self.active_event = None

        if not self.config.enabled:
            return np.zeros(6, dtype=np.float64)

        if step >= self._next_push_step:
            self.active_event = self.sample_random_event(step)
            self.event_history.append(self.active_event)
            self._next_push_step = self._sample_next_interval(step)
            return self.active_event.wrench_6d

        return np.zeros(6, dtype=np.float64)


def sample_disturbance_wrench_jax(
    rng: jax.Array,
    config: DisturbanceConfig,
) -> jax.Array:
    """Sample a 6D external wrench [Fx, Fy, Fz, Tx, Ty, Tz] in JAX for MJX training."""
    if not config.enabled or config.force_magnitude_range[1] <= 0.0:
        return jnp.zeros((6,), dtype=jnp.float32)

    k_mag, k_ang, k_fz, k_tmag, k_tdir = jax.random.split(rng, 5)
    f_min, f_max = config.force_magnitude_range
    mag = jax.random.uniform(k_mag, (), minval=f_min, maxval=f_max)

    # Sample from frontal/diagonal sectors during training so pure +/-90 deg lateral shocks
    # remain held-out unseen directions during evaluation
    a_min, a_max = config.angle_range_rad
    azimuth = jax.random.uniform(k_ang, (), minval=a_min, maxval=a_max)
    fx = mag * jnp.cos(azimuth)
    fy = mag * jnp.sin(azimuth)

    max_fz = mag * config.vertical_force_fraction
    fz = jax.random.uniform(k_fz, (), minval=-max_fz, maxval=max_fz)

    t_min, t_max = config.torque_magnitude_range
    t_mag = jax.random.uniform(k_tmag, (), minval=t_min, maxval=t_max)
    raw_dir = jax.random.normal(k_tdir, (3,))
    unit_dir = raw_dir / (jnp.linalg.norm(raw_dir) + 1e-6)
    torque = t_mag * unit_dir

    return jnp.array([fx, fy, fz, torque[0], torque[1], torque[2]], dtype=jnp.float32)
