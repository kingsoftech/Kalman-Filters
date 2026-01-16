import numpy as np

class KalmanFilter:
    def __init__(self, A, B, C, Q, R, P, x0):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = P  # Current error covariance
        self.x = x0 # Current state estimate
        
    def predict(self, u):
        # 1. Predict State: x = A*x + B*u
        # Ensure u is the correct shape
        u = np.atleast_2d(u).reshape(-1, 1)
        
        self.x = self.A @ self.x + self.B @ u
        
        # 2. Predict Covariance: P = A*P*A.T + Q
        self.P = self.A @ self.P @ self.A.T + self.Q
        
        return self.x

    def update(self, z):
        # Ensure z is the correct shape
        z = np.atleast_2d(z).reshape(-1, 1)
        
        # 1. Calculate Innovation (Residual): y = z - C*x_pred
        y = z - self.C @ self.x
        
        # 2. Calculate Innovation Covariance: S = C*P*C.T + R
        S = self.C @ self.P @ self.C.T + self.R
        
        # 3. Calculate Kalman Gain: K = P*C.T * inv(S)
        # Using linalg.solve is numerically more stable than inv()
        K = self.P @ self.C.T @ np.linalg.inv(S)
        
        # 4. Update State Estimate: x = x_pred + K*y
        self.x = self.x + K @ y
        
        # 5. Update Error Covariance: P = (I - K*C) * P_pred
        I = np.eye(self.A.shape[0])
        self.P = (I - K @ self.C) @ self.P
        
        return self.x

