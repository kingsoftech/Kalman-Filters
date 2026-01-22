import numpy as np

class SrKalmanFilter:
    """
    Implements a Standard Linear Kalman Filter.
    
    Note: Despite the name 'SrKalmanFilter', this implementation currently 
    uses the standard covariance (P) update equations, not the Square Root 
    (Cholesky/QR) formulation.
    
    Attributes:
        A (numpy.ndarray): State Transition Matrix (F in some texts).
        B (numpy.ndarray): Control Input Matrix (G in some texts).
        C (numpy.ndarray): Measurement Matrix (H in some texts).
        Q (numpy.ndarray): Process Noise Covariance Matrix.
        R (numpy.ndarray): Measurement Noise Covariance Matrix.
        P (numpy.ndarray): Error Covariance Matrix.
        x (numpy.ndarray): Current State Estimate vector.
    """

    def __init__(self, A, B, C, Q, R, P, x0):
        """
        Initializes the Kalman Filter with system matrices and initial state.

        Args:
            A: State transition matrix.
            B: Control input matrix.
            C: Measurement matrix (maps state to measurement).
            Q: Process noise covariance.
            R: Measurement noise covariance.
            P: Initial error covariance matrix.
            x0: Initial state estimate.
        """
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x0

    def predict(self, u):
        """
        Performs the Time Update (Prediction) step.
        
        Projects the state and error covariance ahead one time step.
        
        Args:
            u: Control input vector (e.g., motor commands, acceleration).
        """
        # Ensure input u is a column vector
        u = np.atleast_2d(u).reshape(-1, 1)
        
        # 1. Project the state ahead
        # x_k- = A * x_k-1 + B * u_k
        self.x = self.A @ self.x + self.B @ u
        
        # 2. Project the error covariance ahead
        # P_k- = A * P_k-1 * A' + Q
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        """
        Performs the Measurement Update (Correction) step.
        
        Corrects the predicted state using the new measurement z.
        
        Args:
            z: Measurement vector from sensors.
            
        Returns:
            tuple: (Updated State Vector, 3-Sigma Bounds Vector)
        """
        # Ensure measurement z is a column vector
        z = np.atleast_2d(z).reshape(-1, 1)
        
        # 1. Compute the Innovation (residual)
        # y = z - H * x_k-
        y = z - self.C @ self.x
        
        # 2. Compute Innovation Covariance
        # S = H * P_k- * H' + R
        S = self.C @ self.P @ self.C.T + self.R
        
        # 3. Compute Optimal Kalman Gain
        # K = P_k- * H' * inv(S)
        # Note: Using inv() is numerically unstable for large matrices; 
        # in production, solve(S, ...) is preferred.
        K = self.P @ self.C.T @ np.linalg.inv(S)
        
        # 4. Update State Estimate
        # x_k = x_k- + K * y
        self.x = self.x + K @ y
        
        # 5. Update Error Covariance
        # P_k = (I - K * H) * P_k-
        I = np.eye(self.A.shape[0])
        self.P = (I - K @ self.C) @ self.P
        
        # 6. Calculate 3-Sigma Bounds for error visualization
        # Extract diagonal elements (variances), take sqrt (std dev), multiply by 3
        threeSigma = 3 * np.sqrt(np.diag(self.P))
        
        return self.x, threeSigma