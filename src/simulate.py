"""
Simulate a small drone flying a planar figure-of-eight, plus the sensors on it.

Sensors (all noisy):
  - IMU @ 100 Hz : body-frame acceleration (ax, ay, small constant bias) and yaw rate (gyro, constant bias)
  - GNSS @ 5 Hz  : world-frame position (x, y)  -- switched OFF during an outage window
  - Magnetometer @ 10 Hz : heading (yaw)
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class SimConfig:
    duration: float = 120.0        # s
    imu_rate: float = 100.0        # Hz
    gnss_rate: float = 5.0         # Hz
    mag_rate: float = 10.0         # Hz
    accel_noise: float = 0.05      # m/s^2  (1-sigma)
    gyro_noise: float = 0.005      # rad/s
    gyro_bias: float = 0.02        # rad/s  (~1.1 deg/s, estimated by the filter)
    accel_bias: tuple = (0.03, -0.02)  # m/s^2 (NOT modelled -> causes drift in outages)
    filter_accel_noise: float = 0.15   # m/s^2 inflated in the filter to cover the bias
    gnss_noise: float = 1.5        # m
    mag_noise: float = np.deg2rad(3.0)  # rad
    outage_start: float = 50.0     # s  -- GNSS-denied window
    outage_end: float = 80.0       # s
    seed: int = 1


def wrap(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def true_trajectory(t):
    """Figure-of-eight (lemniscate). Returns position, velocity, acceleration (world frame)."""
    A, w = 60.0, 2 * np.pi / 60.0          # 60 m wide, one loop per 60 s
    px = A * np.sin(w * t)
    py = A * np.sin(w * t) * np.cos(w * t)
    vx = A * w * np.cos(w * t)
    vy = A * w * np.cos(2 * w * t)
    ax = -A * w**2 * np.sin(w * t)
    ay = -2 * A * w**2 * np.sin(2 * w * t)
    return np.stack([px, py]), np.stack([vx, vy]), np.stack([ax, ay])


def simulate(cfg: SimConfig = SimConfig()):
    rng = np.random.default_rng(cfg.seed)
    dt = 1.0 / cfg.imu_rate
    t = np.arange(0.0, cfg.duration, dt)

    p, v, a_w = true_trajectory(t)
    yaw = np.arctan2(v[1], v[0])                  # drone flies nose-first
    yaw_rate = np.gradient(np.unwrap(yaw), dt)

    # IMU measures acceleration in the BODY frame: a_b = R(yaw)^T a_w
    c, s = np.cos(yaw), np.sin(yaw)
    a_b = np.stack([c * a_w[0] + s * a_w[1], -s * a_w[0] + c * a_w[1]])

    imu_acc = a_b + np.array(cfg.accel_bias)[:, None] + rng.normal(0, cfg.accel_noise, a_b.shape)
    imu_gyro = yaw_rate + cfg.gyro_bias + rng.normal(0, cfg.gyro_noise, t.shape)

    gnss_every = int(cfg.imu_rate / cfg.gnss_rate)
    mag_every = int(cfg.imu_rate / cfg.mag_rate)
    k = np.arange(len(t))
    in_outage = (t >= cfg.outage_start) & (t < cfg.outage_end)
    gnss_valid = (k % gnss_every == 0) & ~in_outage
    mag_valid = k % mag_every == 0

    gnss = p + rng.normal(0, cfg.gnss_noise, p.shape)
    mag = wrap(yaw + rng.normal(0, cfg.mag_noise, t.shape))

    truth = {"p": p, "v": v, "yaw": yaw, "bias": np.full_like(t, cfg.gyro_bias)}
    sensors = {"acc": imu_acc, "gyro": imu_gyro,
               "gnss": gnss, "gnss_valid": gnss_valid,
               "mag": mag, "mag_valid": mag_valid}
    return t, truth, sensors
