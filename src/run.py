"""
Run the simulation, the EKF and an IMU-only dead-reckoning baseline; save plots + metrics.

    python src/run.py            # default scenario (30 s GNSS outage)
    python src/run.py --no-mag   # see what happens without a heading sensor
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulate import SimConfig, simulate, wrap
from ekf import EKF

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def run_filter(t, truth, sens, cfg, use_gnss=True, use_mag=True):
    dt = t[1] - t[0]
    x0 = [truth["p"][0, 0], truth["p"][1, 0], truth["v"][0, 0], truth["v"][1, 0],
          truth["yaw"][0], 0.0]                       # bias starts unknown (0)
    P0 = np.diag([2.0**2, 2.0**2, 0.5**2, 0.5**2, np.deg2rad(5)**2, 0.05**2])
    ekf = EKF(x0, P0, cfg.filter_accel_noise, cfg.gyro_noise)

    n = len(t)
    X = np.zeros((n, 6)); Psig = np.zeros((n, 6)); nis = []
    for k in range(n):
        if k > 0:
            u = [sens["acc"][0, k - 1], sens["acc"][1, k - 1], sens["gyro"][k - 1]]
            ekf.predict(u, dt)
        if use_gnss and sens["gnss_valid"][k]:
            nis.append(ekf.update_gnss(sens["gnss"][:, k], cfg.gnss_noise))
        if use_mag and sens["mag_valid"][k]:
            ekf.update_mag(sens["mag"][k], cfg.mag_noise)
        X[k] = ekf.x
        Psig[k] = np.sqrt(np.diag(ekf.P))
    return X, Psig, np.array(nis)


def rmse(e):
    return float(np.sqrt(np.mean(e**2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-mag", action="store_true")
    args = ap.parse_args()

    cfg = SimConfig()
    t, truth, sens = simulate(cfg)
    os.makedirs(OUT, exist_ok=True)

    X, S, nis = run_filter(t, truth, sens, cfg, use_mag=not args.no_mag)
    DR, _, _ = run_filter(t, truth, sens, cfg, use_gnss=False, use_mag=False)

    pos_err = np.linalg.norm(X[:, :2].T - truth["p"], axis=0)
    dr_err = np.linalg.norm(DR[:, :2].T - truth["p"], axis=0)
    yaw_err = wrap(X[:, 4] - truth["yaw"])
    gv = sens["gnss_valid"]
    gnss_err = np.linalg.norm(sens["gnss"][:, gv] - truth["p"][:, gv], axis=0)
    outage = (t >= cfg.outage_start) & (t < cfg.outage_end)

    lines = [
        "Metric                                   Value",
        "---------------------------------------  --------",
        f"GNSS-only position RMSE                  {rmse(gnss_err):6.2f} m",
        f"IMU dead-reckoning position RMSE         {rmse(dr_err):6.2f} m",
        f"EKF position RMSE (whole flight)         {rmse(pos_err):6.2f} m",
        f"EKF position RMSE (GNSS available)       {rmse(pos_err[~outage]):6.2f} m",
        f"EKF max position error in 30 s outage    {pos_err[outage].max():6.2f} m",
        f"EKF heading RMSE                         {np.rad2deg(rmse(yaw_err)):6.2f} deg",
        f"Gyro bias: true {np.rad2deg(cfg.gyro_bias):.3f}, "
        f"estimated {np.rad2deg(X[-1, 5]):.3f} deg/s",
        f"Mean GNSS NIS (expect ~2.0 if consistent) {nis[20:].mean():5.2f}",
    ]
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUT, "metrics.txt"), "w") as fh:
        fh.write(report + "\n")

    # ---- plots ----
    shade = dict(color="grey", alpha=0.15, label="GNSS outage")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(*truth["p"], "k-", lw=2, label="Truth")
    ax.plot(*sens["gnss"][:, gv], ".", ms=3, color="tab:orange", alpha=0.6, label="GNSS fixes")
    ax.plot(DR[:, 0], DR[:, 1], "--", color="tab:red", label="IMU dead-reckoning")
    ax.plot(X[:, 0], X[:, 1], color="tab:blue", label="EKF")
    ax.plot(*truth["p"][:, outage], color="grey", lw=8, alpha=0.25, label="Outage segment")
    ax.set_xlim(-90, 90); ax.set_ylim(-60, 60)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("INS/GNSS EKF — figure-of-eight flight with 30 s GNSS outage")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "trajectory.png"), dpi=150)

    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ex = X[:, 0] - truth["p"][0]
    axs[0].plot(t, ex, label="x error"); axs[0].fill_between(t, -3 * S[:, 0], 3 * S[:, 0], alpha=0.25, label="±3σ")
    axs[0].set_ylabel("x err [m]")
    axs[1].plot(t, np.rad2deg(yaw_err), label="yaw error")
    axs[1].fill_between(t, -3 * np.rad2deg(S[:, 4]), 3 * np.rad2deg(S[:, 4]), alpha=0.25, label="±3σ")
    axs[1].set_ylabel("yaw err [deg]")
    axs[2].plot(t, np.rad2deg(X[:, 5]), label="estimated")
    axs[2].axhline(np.rad2deg(cfg.gyro_bias), color="k", ls="--", label="true")
    axs[2].set_ylabel("gyro bias [deg/s]"); axs[2].set_xlabel("time [s]")
    for a in axs:
        a.axvspan(cfg.outage_start, cfg.outage_end, **shade); a.grid(alpha=0.3); a.legend(fontsize=8, loc="upper right")
    axs[0].set_title("Estimation error with 3σ covariance bounds")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "errors.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.semilogy(t, dr_err, "--", color="tab:red", label="IMU dead-reckoning")
    ax.semilogy(t, pos_err, color="tab:blue", label="EKF")
    ax.axhline(cfg.gnss_noise, color="tab:orange", ls=":", label="GNSS 1σ noise")
    ax.axvspan(cfg.outage_start, cfg.outage_end, **shade)
    ax.set_xlabel("time [s]"); ax.set_ylabel("position error [m]"); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8); ax.set_title("Position error: fusion vs dead-reckoning")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "position_error.png"), dpi=150)
    print(f"Plots saved to {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
