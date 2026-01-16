import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import Kalmanfilter as kf   
import mass_spring_dampersystem


    # Initialize System
msds = mass_spring_dampersystem.MassSpringDamperSystem(mass=50.0, spring_constant=2.0, damping_coefficient=4.0, dt=0.1)

# Extract Model Matrices
# Note: We must call .models() to get the matrices, they are not attributes of msds
Ad, Bd, Cd, Qd, Rd = msds.models()
x0, P0 = msds.initial_state()

# Generate Input and Run Simulation (Ground Truth)
duration = 100.0  # seconds
F, t_hist = msds.systems_input(duration)
x_hist, z_hist = msds.simulation(F, x0, Ad, Bd, Cd, Qd, Rd)

# Initialize Kalman Filter with the same model matrices
init_kalman = kf.KalmanFilter(
    A=Ad, B=Bd, C=Cd, Q=Qd, R=Rd, P=P0, x0=x0
)

# Run Estimation
num_steps = x_hist.shape[1]
x_est_hist = np.zeros((2, num_steps))
threeSigma_hist = np.zeros((2, num_steps))  
x_est_hist[:, 0:1] = x0 

for k in range(num_steps - 1):
    # 1. Predict (using input u at time k)
    u_k = F[:, k]
    init_kalman.predict(u_k)
    
    # 2. Update (using measurement z at time k+1)
    # We use z_hist[:, k+1] because prediction moves us to k+1
    z_next = z_hist[:, k+1] 
    x_est, threeSigma = init_kalman.update(z_next)
    
    # Store
    x_est_hist[:, k+1:k+2] = x_est
    threeSigma_hist[:, k+1:k+2] = threeSigma.reshape(-1, 1)

# [Assuming previous code has run and we have x_hist, x_est_hist, threeSigma_hist, t_hist]

# 1. Calculate Errors
error_hist = x_hist - x_est_hist



# --- Figure 1: States and Estimates with Bounds ---
plt.figure(figsize=(12, 6))

# Define colors for clarity (State 1 = Blue, State 2 = Orange/Red)
colors = ['b', 'orange']
labels_true = ['True Position', 'True Velocity']
labels_est = ['Position Estimate', 'Velocity Estimate']

# Loop through states to plot them all on one figure (like your MATLAB code)
for k in range(2): 
    # 1. Plot True State (Solid Line)
    plt.plot(t_hist, x_hist[k, :], color=colors[k], linestyle='-', linewidth=1.5, label=labels_true[k])
    
    # 2. Plot Estimate (Dashed Line)
    plt.plot(t_hist, x_est_hist[k, :], color=colors[k], linestyle='--', linewidth=1.5, label=labels_est[k])
    
    # 3. Plot Bounds (Shaded Region)
    # Python's fill_between replaces the complex [t fliplr(t)] construction
    plt.fill_between(t_hist, 
                     x_est_hist[k, :] - threeSigma_hist[k, :], 
                     x_est_hist[k, :] + threeSigma_hist[k, :], 
                     color=colors[k], alpha=0.2, linewidth=0, label=f'Bounds State {k+1}')

plt.title('Demonstration of Kalman filter state estimates')
plt.xlabel('Time (s)')
plt.ylabel('State (m or m/s)')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# --- Figure 2: Estimation Errors ---
plt.figure(figsize=(10, 8))
xerr = x_hist - x_est_hist  # Calculate Error
nx = x_hist.shape[0]        # Number of states (2)

for k in range(nx):
    plt.subplot(nx, 1, k + 1)
    
    # 1. Plot Bounds centered at 0
    # Equivalent to: fill(t2, [-bound fliplr(bound)], ...)
    plt.fill_between(t_hist, 
                     -threeSigma_hist[k, :], 
                     threeSigma_hist[k, :], 
                     color='blue', alpha=0.25, linewidth=0, label='3$\sigma$ Bound')
    
    # 2. Plot Error
    plt.plot(t_hist, xerr[k, :], 'b', linewidth=1.5, label='Error')
    
    plt.title(f'State {k+1} estimation error with bounds')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.grid(True)
    plt.legend(loc='upper right')

plt.tight_layout()
plt.show()