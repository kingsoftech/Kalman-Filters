import numpy as np
from scipy.linalg import expm

class MassSpringDamperSystem:
    def __init__(self, mass, spring_constant, damping_coefficient, dt=0.1):
        self.mass = mass
        self.spring_constant = spring_constant
        self.damping_coefficient = damping_coefficient
        self.dt = dt # Move time step to init so it's accessible everywhere

    def models(self):
        # 1. Continuous System Matrices
        A = np.array([[0, 1],
                      [-self.spring_constant / self.mass, -self.damping_coefficient / self.mass]])
        B = np.array([[0],
                      [1 / self.mass]])
        C = np.array([[1, 0]]) # Output matrix (Position)
        D = np.array([[0]])
        
        # 2. Continuous Process Noise Covariance (Qc)
        # Bw maps noise to state. Assuming input noise affects acceleration.
        Bw = np.array([[0],
                       [1 / self.mass]])
        Sw = 0.01      # Continuous Process noise std dev
        Qc = Bw @ Bw.T * (Sw**2) # Continuous Covariance Matrix

        # 3. Discretization of System (Zero Order Hold)
        Ad = expm(A * self.dt)
        # Bd calculation: Integral(e^At)*B. 
        # Note: If A is singular, this inverse method fails, but for MSD it's usually fine.
        Bd = np.linalg.inv(A) @ (Ad - np.eye(2)) @ B

        # 4. Discretization of Noise Covariance (Van Loan's Method)
        # Construct the Hamiltonian-like matrix
        # Z = [-A  Qc]
        #     [ 0  A.T]
        zeros_block = np.zeros_like(A) # Fixed: Must be 2x2, not 1x2
        
        # Using block helper for cleaner matrix construction
        Z = np.block([
            [-A, Qc],
            [zeros_block, A.T]
        ]) * self.dt

        Phi = expm(Z) # The big matrix exponential
        
        # Extract blocks from Phi
        # Phi_12 is top-right (rows 0:2, cols 2:4)
        # Phi_22 is bottom-right (rows 2:4, cols 2:4)
        Phi_12 = Phi[0:2, 2:4]
        Phi_22 = Phi[2:4, 2:4]
        
        # Discrete Process Noise Covariance Qd = Phi_22.T * Phi_12
        sigmalw = Phi_22.T @ Phi_12
        
        # Ensure symmetry (numerical stability)
        sigmalw = (sigmalw + sigmalw.T) / 2

        # 5. Measurement Noise Covariance (Discrete)
        Sv = 0.00001 # Measurement noise std dev
        # For discrete measurement noise, R = Sv^2 / dt (assuming band-limited)
        # We assume result needs to be 2D (1x1) for Cholesky
        sigmalv = np.array([[ (Sv**2) / self.dt ]])

        return Ad, Bd, C, D, sigmalw, sigmalv

    def initial_state(self):
        x0 = np.array([[0],
                       [0]]) # Initial position and velocity (2x1)
        P0 = np.eye(2)       # Initial covariance
        z0 = np.array([[0]]) # Initial measurement
        return x0, P0, z0
    
    def systems_input(self, duration):
        # Fixed: Generate time vector based on self.dt
        t = np.arange(0, duration, self.dt)
        # Reshape t to be (1, N) for matrix math
        t_row = t.reshape(1, -1)
        
        # Example: sinusoidal force
        frequency = 0.1 # Hz
        F = 10 * np.sin(2 * np.pi * frequency * t_row)
        return F, t
    
    def simulation(self):
        Ad, Bd, C, D, sigmalw, sigmalv = self.models()
        x0, P0, z0 = self.initial_state()
        
        # Define simulation duration (seconds)
        duration = 100 
        F, t = self.systems_input(duration)

        num_steps = F.shape[1]
        
        # Initialize storage
        x = np.zeros((2, num_steps))
        z = np.zeros((1, num_steps))
        
        # Set initial conditions
        x[:, 0:1] = x0
        # z[:, 0] is calculated in loop, or set based on x0
        z[:, 0:1] = C @ x0

        # Pre-compute Cholesky for noise injection
        # Use simple try/except or slight diagonal boost if not Positive Definite
        try:
            L_w = np.linalg.cholesky(sigmalw)
        except np.linalg.LinAlgError:
            # Fallback for numerical zeros
            L_w = np.zeros_like(sigmalw)
            
        L_v = np.linalg.cholesky(sigmalv)

        # Simulation Loop
        for k in range(0, num_steps - 1):
            # Generate Noise
            w_k = L_w @ np.random.randn(2, 1)
            v_k = L_v @ np.random.randn(1, 1)
            
            # Input at time k
            u_k = F[:, k:k+1]
            
            # State Update: x[k+1] = Ad*x[k] + Bd*u[k] + w[k]
            x[:, k+1:k+2] = Ad @ x[:, k:k+1] + Bd @ u_k + w_k
            
            # Measurement Update: z[k+1] = C*x[k+1] + ... + v[k+1]
            # (Note: Measurement usually taken at current step. Here calculating for next step)
            z[:, k+1:k+2] = C @ x[:, k+1:k+2] + D @ u_k + v_k

        return x, z, t

# --- Execution ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create Instance
    msd = MassSpringDamperSystem(mass=1.0, spring_constant=2.0, damping_coefficient=0.5)
    
    # Run Simulation
    x_hist, z_hist, t_hist = msd.simulation()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t_hist, x_hist[0, :], label='Position (State)')
    plt.plot(t_hist, z_hist[0, :], 'r.', markersize=2, alpha=0.3, label='Noisy Measurement')
    plt.title('Mass-Spring-Damper Simulation')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t_hist, x_hist[1, :], label='Velocity', color='orange')
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.show()