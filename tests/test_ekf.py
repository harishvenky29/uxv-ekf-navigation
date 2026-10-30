import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ekf import EKF            # noqa: E402
from simulate import SimConfig, simulate, wrap  # noqa: E402
from run import run_filter     # noqa: E402


def numerical_jacobian(fun, x, eps=1e-6):
    fx = fun(x)
    J = np.zeros((len(fx), len(x)))
    for i in range(len(x)):
        dx = np.zeros_like(x); dx[i] = eps
        diff = fun(x + dx) - fx
        diff[4] = wrap(diff[4])
        J[:, i] = diff / eps
    return J


def test_state_jacobian_matches_numerical():
    x = np.array([1.0, -2.0, 3.0, 0.5, 0.7, 0.01])
    u = np.array([0.4, -0.3, 0.2])
    dt = 0.01
    J_num = numerical_jacobian(lambda s: EKF.f(s, u, dt), x)
    assert np.allclose(EKF.F_jacobian(x, u, dt), J_num, atol=1e-6)


def test_input_jacobian_matches_numerical():
    x = np.array([1.0, -2.0, 3.0, 0.5, 0.7, 0.01])
    u = np.array([0.4, -0.3, 0.2])
    dt = 0.01
    J_num = numerical_jacobian(lambda uu: EKF.f(x, uu, dt), u)
    assert np.allclose(EKF.G_jacobian(x, dt), J_num, atol=1e-6)


def test_filter_beats_gnss_and_learns_bias():
    cfg = SimConfig()
    t, truth, sens = simulate(cfg)
    X, _, nis = run_filter(t, truth, sens, cfg)
    outage = (t >= cfg.outage_start) & (t < cfg.outage_end)
    err = np.linalg.norm(X[:, :2].T - truth["p"], axis=0)
    assert np.sqrt(np.mean(err[~outage] ** 2)) < cfg.gnss_noise   # fusion beats raw GNSS
    assert abs(X[-1, 5] - cfg.gyro_bias) < 0.005                    # bias observed
    assert 1.0 < nis[20:].mean() < 3.0                              # filter is consistent
