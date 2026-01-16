import numpy as np
from scipy.linalg import expm
# --- 1. System Model Class ---
class MassSpringDamperSystem:
    def __init__(self, mass, spring_constant, damping_coefficient, dt=0.1):
        self.mass = mass
        self.spring_constant = spring_constant
        self.damping_coefficient = damping_coefficient
        self.dt = dt

    def models(self):
        # Continuous Matrices
        A = np.array([[0, 1], 
                      [-self.spring_constant/self.mass, -self.damping_coefficient/self.mass]])
        B = np.array([[0], 
                      [1/self.mass]])
        C = np.array([[1, 0]]) # Measure Position
        
        # Process Noise (Continuous)
        Bw = np.array([[0], [1/self.mass]])
        Sw = 0.01 
        Qc = Bw @ Bw.T * (Sw**2)

        # Discretization of Noise Covariance (Van Loan's Method)
        zeros = np.zeros_like(A)
        Z = np.block([[-A, Qc], 
                      [zeros, A.T]]) * self.dt
        Phi = expm(Z)
        Phi_12 = Phi[0:2, 2:4]
        Phi_22 = Phi[2:4, 2:4]
        
        Qd = Phi_22.T @ Phi_12 # Discrete Process Noise Covariance
        
        # Discrete Measurement Noise
        Sv = 0.00001
        Rd = np.array([[Sv * self.dt]]) 

        # System Discretization
        Ad = expm(A * self.dt)
        Bd = np.linalg.inv(A) @ (Ad - np.eye(2)) @ B

        return Ad, Bd, C, Qd, Rd

    def initial_state(self):
        x0 = np.array([[0], [0]])
        P0 = np.eye(2)
        return x0, P0

    def systems_input(self, duration):
        t = np.arange(0, duration, self.dt)
        F = 10 * np.sin(2 * np.pi * 0.1 * t).reshape(1, -1)
        return F, t

    def simulation(self, F, x0, Ad, Bd, C, Qd, Rd):
        num_steps = F.shape[1]
        x = np.zeros((2, num_steps))
        z = np.zeros((1, num_steps))
        
        # 1. Set Initial State
        x[:, 0:1] = x0
        
        # 2. Pre-compute Noise Matrices
        try:
            L_w = np.linalg.cholesky(Qd)
        except np.linalg.LinAlgError:
            L_w = np.zeros_like(Qd)
            
        L_v = np.linalg.cholesky(Rd)

        # 3. Generate Initial Measurement (The Fix)
        # z0 = C*x0 + v0
        v0 = L_v @ np.random.randn(1, 1)
        z[:, 0:1] = C @ x0 + v0

        # 4. Loop
        for k in range(num_steps - 1):
            w = L_w @ np.random.randn(2, 1)
            v = L_v @ np.random.randn(1, 1)
            
            # State Update (k -> k+1)
            # Use current input F[:, k]
            u_current = F[:, k:k+1] 
            x[:, k+1:k+2] = Ad @ x[:, k:k+1] + Bd @ u_current + w
            
            # Measurement Update (at k+1)
            z[:, k+1:k+2] = C @ x[:, k+1:k+2] + v
            
        return x, z
