import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import SrKalmanfilter as kf
from imu_roll_pitch import IMUSystem

# ==========================================
# 1. THE KALMAN FILTER CLASS
# ==========================================
# ==========================================
# 2. THE SYSTEM MODEL (6-AXIS IMU)
# ==========================================

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

# Initialize System
imu = IMUSystem(dt=0.01)
Ad, Bd, Cd, Qd, Rd = imu.models()
x0, P0 = imu.initial_state()




# Generate Ground Truth
duration = 10.0
U, t_hist = imu.systems_input(duration)
x_hist, z_hist = imu.simulation(U, x0, Ad, Bd, Cd, Qd, Rd)

# Initialize Kalman Filter
kf = kf.SrKalmanFilter(A=Ad, B=Bd, C=Cd, Q=Qd, R=Rd, P=P0, x0=x0)

# Run Estimation Loop
num_steps = x_hist.shape[1]
x_est_hist = np.zeros((4, num_steps))
threeSigma_hist = np.zeros((4, num_steps))
x_est_hist[:, 0:1] = x0

for k in range(num_steps - 1):
    # Predict (Using Gyro Input)
    u_k = U[:, k:k+1]
    kf.predict(u_k)
    
    # Update (Using Accel Measurement)
    z_next = z_hist[:, k+1:k+2]
    x_est, threeSigma = kf.update(z_next)
    
    # Store
    x_est_hist[:, k+1:k+2] = x_est
    threeSigma_hist[:, k+1:k+2] = threeSigma.reshape(-1, 1)


# ==========================================
# 4. PLOTTING
# ==========================================


# --- Figure 1: Roll Angle Tracking ---
plt.figure(figsize=(12, 6))
state_idx = 0 # 0 = Roll, 1 = Pitch
plt.plot(t_hist, z_hist[state_idx, :], 'k.', markersize=2, alpha=0.3, label='Noisy Accel Measurements')
plt.plot(t_hist, x_hist[state_idx, :], 'g-', linewidth=2, label='True Roll Angle')
plt.plot(t_hist, x_est_hist[state_idx, :], 'b--', linewidth=2, label='Estimated Roll Angle')

# Plot Bounds
bound_upper = x_est_hist[state_idx, :] + threeSigma_hist[state_idx, :]
bound_lower = x_est_hist[state_idx, :] - threeSigma_hist[state_idx, :]
plt.fill_between(t_hist, bound_lower, bound_upper, color='blue', alpha=0.2, label='3-Sigma Confidence')

plt.title('Kalman Filter Performance: Roll Angle')
plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# --- Figure 2: Bias Estimation (The Hidden State) ---
# This is crucial for IMUs - checking if the filter learned the bias
plt.figure(figsize=(12, 6))
bias_idx = 2 # 2 = Roll Bias
plt.plot(t_hist, x_hist[bias_idx, :], 'g-', linewidth=2, label='True Gyro Bias (Random Walk)')
plt.plot(t_hist, x_est_hist[bias_idx, :], 'r--', linewidth=2, label='Estimated Gyro Bias')

bound_upper = x_est_hist[bias_idx, :] + threeSigma_hist[bias_idx, :]
bound_lower = x_est_hist[bias_idx, :] - threeSigma_hist[bias_idx, :]
plt.fill_between(t_hist, bound_lower, bound_upper, color='red', alpha=0.2, label='3-Sigma Confidence')

plt.title('Hidden State Estimation: Gyro Bias Learning')
plt.xlabel('Time (s)')
plt.ylabel('Bias (deg/s)')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# --- Figure 3: Estimation Error Analysis ---
plt.figure(figsize=(10, 8))
xerr = x_hist - x_est_hist
titles = ['Roll Error', 'Pitch Error', 'Roll Bias Error', 'Pitch Bias Error']

for k in range(4):
    plt.subplot(4, 1, k + 1)
    
    # Plot Error
    plt.plot(t_hist, xerr[k, :], 'b', linewidth=1.5)
    
    # Plot Bounds
    plt.plot(t_hist, threeSigma_hist[k, :], 'r--', linewidth=1)
    plt.plot(t_hist, -threeSigma_hist[k, :], 'r--', linewidth=1)
    
    plt.title(titles[k])
    plt.ylabel('Error')
    plt.grid(True)

plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()