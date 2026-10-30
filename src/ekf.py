"""
Loosely-coupled INS/GNSS Extended Kalman Filter (2D).

State  x = [px, py, vx, vy, yaw, b_g]        (position, velocity, heading, gyro bias)
Input  u = [ax_b, ay_b, w_meas]              (IMU: body accel + gyro rate)

Predict (IMU, 100 Hz):
    yaw' = yaw + (w_meas - b_g) dt
    a_w  = R(yaw) a_b
    v'   = v + a_w dt
    p'   = p + v dt + 0.5 a_w dt^2
    b_g' = b_g                       (random walk)

Update:
    GNSS  z = [px, py]   -> H = [I2 0 0 0 0]
    Mag   z = yaw        -> H = [0 0 0 0 1 0]   (innovation angle-wrapped)
"""
import numpy as np

from simulate import wrap


class EKF:
    N = 6

    def __init__(self, x0, P0, accel_noise, gyro_noise, bias_walk=1e-4):
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        self.q_acc = accel_noise**2
        self.q_gyro = gyro_noise**2
        self.q_bias = bias_walk**2

    # ---------- model ----------
    @staticmethod
    def f(x, u, dt):
        px, py, vx, vy, yaw, bg = x
        ax_b, ay_b, w = u
        c, s = np.cos(yaw), np.sin(yaw)
        ax = c * ax_b - s * ay_b
        ay = s * ax_b + c * ay_b
        return np.array([
            px + vx * dt + 0.5 * ax * dt**2,
            py + vy * dt + 0.5 * ay * dt**2,
            vx + ax * dt,
            vy + ay * dt,
            wrap(yaw + (w - bg) * dt),
            bg,
        ])

    @staticmethod
    def F_jacobian(x, u, dt):
        """d f / d x."""
        yaw = x[4]
        ax_b, ay_b, _ = u
        c, s = np.cos(yaw), np.sin(yaw)
        # derivative of world accel wrt yaw
        dax = -s * ax_b - c * ay_b
        day = c * ax_b - s * ay_b
        F = np.eye(6)
        F[0, 2] = dt
        F[1, 3] = dt
        F[0, 4] = 0.5 * dax * dt**2
        F[1, 4] = 0.5 * day * dt**2
        F[2, 4] = dax * dt
        F[3, 4] = day * dt
        F[4, 5] = -dt
        return F

    @staticmethod
    def G_jacobian(x, dt):
        """d f / d u  (maps IMU noise into state noise)."""
        c, s = np.cos(x[4]), np.sin(x[4])
        R = np.array([[c, -s], [s, c]])
        G = np.zeros((6, 3))
        G[0:2, 0:2] = 0.5 * R * dt**2
        G[2:4, 0:2] = R * dt
        G[4, 2] = dt
        return G

    # ---------- filter steps ----------
    def predict(self, u, dt):
        F = self.F_jacobian(self.x, u, dt)
        G = self.G_jacobian(self.x, dt)
        Qu = np.diag([self.q_acc, self.q_acc, self.q_gyro])
        Q = G @ Qu @ G.T
        Q[5, 5] += self.q_bias * dt
        self.x = self.f(self.x, u, dt)
        self.P = F @ self.P @ F.T + Q

    def _update(self, y, H, R):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[4] = wrap(self.x[4])
        # Joseph form keeps P symmetric & positive-definite
        I_KH = np.eye(self.N) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        nis = float(y @ np.linalg.solve(S, y))
        return nis

    def update_gnss(self, z, sigma):
        H = np.zeros((2, 6)); H[0, 0] = H[1, 1] = 1.0
        y = z - H @ self.x
        return self._update(y, H, np.eye(2) * sigma**2)

    def update_mag(self, z, sigma):
        H = np.zeros((1, 6)); H[0, 4] = 1.0
        y = np.array([wrap(z - self.x[4])])
        return self._update(y, H, np.array([[sigma**2]]))
