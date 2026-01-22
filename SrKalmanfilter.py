import numpy as np
class SrKalmanFilter:
    def __init__(self, A, B, C, Q, R, P, x0):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x0

    def predict(self, u):
        u = np.atleast_2d(u).reshape(-1, 1)
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        z = np.atleast_2d(z).reshape(-1, 1)
        y = z - self.C @ self.x
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.A.shape[0])
        self.P = (I - K @ self.C) @ self.P
        threeSigma = 3 * np.sqrt(np.diag(self.P))
        return self.x, threeSigma
