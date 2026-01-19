import numpy as np
from scipy.linalg import expm

# --- 1. System Model Class ---
class IMUSystem:
    def __init__(self, dt=0.01, sigma_gyro=0.1, sigma_bias=0.01, sigma_acc=1.0):
        self.dt = dt
        self.sigma_gyro = sigma_gyro   # deg/s
        self.sigma_bias = sigma_bias   # deg/s^2 (Random Walk)
        self.sigma_acc = sigma_acc     # g or deg (Measurement Noise)

    def models(self):
        # --- Continuous Matrices (Physics) ---
        # State: [phi, theta, b_phi, b_theta]
        # Input: [omega_x, omega_y]
        
        # A matrix: deriv(angle) = -bias
        A = np.array([
            [0, 0, -1, 0],
            [0, 0,  0, -1],
            [0, 0,  0,  0],
            [0, 0,  0,  0]
        ])
        
        # B matrix: deriv(angle) = input
        B = np.array([
            [1, 0],
            [0, 1],
            [0, 0],
            [0, 0]
        ])
        
        # C matrix: We measure [phi, theta] directly (derived from accel)
        C = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ]) 
        
        # --- Process Noise (Continuous) ---
        # Noise enters on:
        # 1. The Angle (Gyro White Noise) -> Rows 0,1
        # 2. The Bias (Bias Instability)  -> Rows 2,3
        
        # We construct G (Noise Input Matrix) mapping noise sources to states
        G = np.eye(4) 
        
        # Spectral Density Matrix (Continuous Q)
        # Power Spectral Density (PSD) ~ sigma^2
        Qc = np.diag([
            self.sigma_gyro**2, 
            self.sigma_gyro**2, 
            self.sigma_bias**2, 
            self.sigma_bias**2
        ])

        # --- Van Loan's Method for Discretization ---
        # We build a large block matrix to solve for Ad, Bd, and Qd all at once.
        # Structure:
        # [-A  G*Qc*G.T  0 ]
        # [ 0     A.T    0 ]  (This part gives Qd)
        # [ 0     B.T    0 ]  (This part helps Bd - simplified here for stability)
        
        # 1. Discretize A and Q (Van Loan)
        n = 4 # State dimension
        zeros = np.zeros((n, n))
        
        # Z matrix for Q discretization
        Z = np.block([
            [-A, G @ Qc @ G.T],
            [zeros, A.T]
        ]) * self.dt
        
        Phi = expm(Z)
        
        Phi_11 = Phi[0:n, 0:n] # This is inv(Ad) technically
        Phi_12 = Phi[0:n, n:2*n]
        Phi_22 = Phi[n:2*n, n:2*n] # This is Ad.T
        
        Ad = Phi_22.T  # Discrete A
        Qd = Phi_22.T @ Phi_12 # Discrete Process Noise
        
        # 2. Discretize B (Robust Method)
        # Since A is singular, we cannot use A^-1. 
        # We use the Taylor series expansion: Bd = (I*dt + A*dt^2/2 + ...) B
        # For our specific A (nilpotent), this simplifies exactly to:
        Bd = (np.eye(n)*self.dt + (A @ np.eye(n) * self.dt**2)/2) @ B
        
        # --- Discrete Measurement Noise ---
        # R is simple: Variance / dt is a common approximation, 
        # but for discrete sensors we usually just use sigma^2 directly.
        Rd = np.eye(2) * (self.sigma_acc**2)

        return Ad, Bd, C, Qd, Rd

    def initial_state(self):
        # [Roll, Pitch, Bias_Roll, Bias_Pitch]
        x0 = np.zeros((4, 1))
        P0 = np.eye(4) * 0.1
        return x0, P0

    def systems_input(self, duration):
        # Simulate a robot rocking back and forth
        t = np.arange(0, duration, self.dt)
        
        # Input (Gyro): Sine wave (rocking)
        # Omega = derivative of sine(angle) = cosine
        freq = 0.5 # Hz
        omega_x = 10 * np.cos(2 * np.pi * freq * t) 
        omega_y = 5 * np.sin(2 * np.pi * freq * t)
        
        u = np.vstack((omega_x, omega_y))
        return u, t

    def simulation(self, U, x0, Ad, Bd, C, Qd, Rd):
        num_steps = U.shape[1]
        x = np.zeros((4, num_steps)) # True State
        z = np.zeros((2, num_steps)) # Measurements
        
        # 1. Set Initial State
        x[:, 0:1] = x0
        
        # 2. Pre-compute Noise Matrices
        try:
            L_w = np.linalg.cholesky(Qd)
        except np.linalg.LinAlgError:
            # Fallback for numerical stability if Qd is tiny
            L_w = np.zeros_like(Qd)
            
        L_v = np.linalg.cholesky(Rd)

        # 3. Generate Initial Measurement
        v0 = L_v @ np.random.randn(2, 1)
        z[:, 0:1] = C @ x0 + v0

        # 4. Loop
        for k in range(num_steps - 1):
            w = L_w @ np.random.randn(4, 1) # Process Noise
            v = L_v @ np.random.randn(2, 1) # Measurement Noise
            
            # State Update (k -> k+1)
            u_current = U[:, k:k+1]
            x[:, k+1:k+2] = Ad @ x[:, k:k+1] + Bd @ u_current + w
            
            # Measurement Update (at k+1)
            # z = C*x + v
            z[:, k+1:k+2] = C @ x[:, k+1:k+2] + v
            
        return x, z

# --- Usage Example ---



