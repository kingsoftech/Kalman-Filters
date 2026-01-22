import numpy as np
import scipy.linalg as la

class SrKalmanFilter:
    """
    Implements a Square Root Kalman Filter (SR-KF) using QR Decomposition.
    
    This improves numerical stability on embedded systems by tracking the 
    Cholesky factor (S) of the covariance matrix P = S @ S.T, rather than P itself.
    """

    def __init__(self, A, B, C, Q, R, P, x0):
        """
        Initializes the SR-KF. Converts P, Q, R to their Cholesky factors.
        """
        self.A = A
        self.B = B
        self.C = C
        
        # 1. Convert Covariances to Square Roots (Lower Triangular)
        # S @ S.T = P
        self.S = np.linalg.cholesky(P)
        self.Sq = np.linalg.cholesky(Q)
        self.Sr = np.linalg.cholesky(R)
        
        self.x = x0
        
        # Store dimensions for block matrix operations
        self.n_states = A.shape[0]
        self.n_meas = C.shape[0]

    def predict(self, u):
        """
        Time Update using QR Decomposition.
        """
        u = np.atleast_2d(u).reshape(-1, 1)
        
        # 1. State Prediction (Standard)
        self.x = self.A @ self.x + self.B @ u
        
        # 2. Covariance Prediction (Square Root Method)
        # We construct the pre-array A:
        # [ (A * S_prev).T ]
        # [    Sq.T        ]
        # We want to find the upper triangular matrix R_qr such that A = Q * R_qr
        # The new S_pred is then R_qr.T (to make it lower triangular)
        
        pre_array = np.vstack([
            (self.A @ self.S).T,
            self.Sq.T
        ])
        
        # Perform QR decomposition
        # mode='r' returns only the R matrix (upper triangular)
        _, R_qr = np.linalg.qr(pre_array)
        
        # The new predicted S is the transpose of the R result
        self.S = R_qr.T

    def update(self, z):
        """
        Measurement Update using QR Decomposition.
        """
        z = np.atleast_2d(z).reshape(-1, 1)
        
        # 1. Construct the Pre-Array for Measurement Update
        # Structure:
        # [ Sr.T        0   ]
        # [ (C * S).T   S.T ]
        
        # Top block
        top_left = self.Sr.T
        top_right = np.zeros((self.n_meas, self.n_states))
        
        # Bottom block
        bot_left = (self.C @ self.S).T
        bot_right = self.S.T
        
        # Combine into one matrix
        pre_array = np.block([
            [top_left, top_right],
            [bot_left, bot_right]
        ])
        
        # 2. Perform QR Decomposition
        # The result R_qr will be Upper Triangular:
        # [ Se.T      K_bar.T ]
        # [ 0         S_new.T ]
        _, R_qr = np.linalg.qr(pre_array)
        
        # 3. Extract Blocks from R_qr
        # Se_T is the square root of the innovation covariance (transposed)
        Se_T = R_qr[0:self.n_meas, 0:self.n_meas]
        
        # K_bar_T is an intermediate term for the gain
        K_bar_T = R_qr[0:self.n_meas, self.n_meas:]
        
        # S_new_T is the square root of the UPDATED covariance (transposed)
        S_new_T = R_qr[self.n_meas:, self.n_meas:]
        
        # Update our internal covariance square root immediately
        self.S = S_new_T.T
        
        # 4. Calculate Kalman Gain
        # From the algebra: K = K_bar * inv(Se)
        # In terms of our transposed blocks: K = (K_bar_T).T * inv(Se_T.T)
        # We can solve this efficiently using triangular solvers
        K_bar = K_bar_T.T
        Se = Se_T.T
        
        # Solve for K: K * Se.T = K_bar  =>  K = K_bar / Se.T
        # We use solve_triangular for speed and stability
        K = la.solve_triangular(Se_T, K_bar.T, lower=False).T

        # 5. Update State
        y = z - self.C @ self.x  # Innovation
        self.x = self.x + K @ y
        
        # 6. Calculate 3-Sigma Bounds (Row Norm Method)
        # Since S is lower triangular, the variance of the i-th state is 
        # the sum of squares of the i-th row of S.
        
        # Square every element, sum along rows (axis=1), then sqrt
        sigmas = np.sqrt(np.sum(self.S**2, axis=1))
        threeSigma = 3 * sigmas
        
        return self.x, threeSigma