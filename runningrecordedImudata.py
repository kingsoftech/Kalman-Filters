import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import pandas as pd
# Assuming you have these saved in separate files or defined above
import Kalmanfilter as kf           
from imu_roll_pitch import IMUSystem 
import matplotlib.gridspec as gridspec
from scipy.stats import norm

# ==========================================
# 1. LOAD & PREPARE REAL DATA
# ==========================================
filename = 'imu_data.csv'
try:
    data = pd.read_csv(filename)
    print(f"Loaded {len(data)} samples from {filename}")
except FileNotFoundError:
    print(f"Error: {filename} not found.")
    exit()

# Extract Columns (ax, ay, az, gx, gy, gz)
ax = data['ax'].values
ay = data['ay'].values
az = data['az'].values
gx = data['gx'].values
gy = data['gy'].values

# --- CRITICAL: Convert Accel to Angles (Measurement z) ---
# Roll (phi)
roll_meas = np.arctan2(ay, az) * (180 / np.pi)
# Pitch (theta)
pitch_meas = np.arctan2(-ax, np.sqrt(ay**2 + az**2)) * (180 / np.pi)

# Setup Time Vector
dt = 0.01
num_steps = len(data)
t_vec = np.arange(num_steps) * dt

# Stack Inputs and Measurements for the loop
# u_hist: [Gyro_X, Gyro_Y]
u_hist = np.vstack((gx, gy))
# z_hist: [Roll_Meas, Pitch_Meas]
z_hist = np.vstack((roll_meas, pitch_meas))

# ==========================================
# 2. INITIALIZE FILTER
# ==========================================
imu = IMUSystem(dt=dt)
Ad, Bd, Cd, Qd, Rd = imu.models()
x0, P0 = imu.initial_state()

# Initialize State with the first measurement to prevent startup spike
x0[0,0] = z_hist[0,0] # Roll
x0[1,0] = z_hist[1,0] # Pitch

# Create Filter Object
my_kf = kf.KalmanFilter(A=Ad, B=Bd, C=Cd, Q=Qd, R=Rd, P=P0, x0=x0)

# Storage for results
x_est_hist = np.zeros((4, num_steps))
threeSigma_hist = np.zeros((4, num_steps))

# Save initial state
x_est_hist[:, 0:1] = x0

# ==========================================
# 3. RUN ESTIMATION LOOP
# ==========================================
print("Running Kalman Filter on Real Data...")

for k in range(num_steps - 1):
    # 1. Predict (Using Gyro Input from CSV)
    # Note: We use u_hist, NOT the undefined 'U'
    u_k = u_hist[:, k:k+1]
    my_kf.predict(u_k)
    
    # 2. Update (Using Accel Measurement from CSV)
    z_next = z_hist[:, k+1:k+2]
    x_est, threeSigma = my_kf.update(z_next)
    for i in range(2): # For both Roll and Pitch
        while x_est[i, 0] > 180: x_est[i, 0] -= 360
        while x_est[i, 0] < -180: x_est[i, 0] += 360

    # 3. Store
    x_est_hist[:, k+1:k+2] = x_est
    threeSigma_hist[:, k+1:k+2] = threeSigma.reshape(-1, 1)



# Create a time vector for plotting
t = np.arange(num_steps) * dt

# --- FIGURE 1: TRACKING PERFORMANCE ---
# Shows how well the filter blends the sensors
fig1 = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 1)

# Plot Roll
ax1 = fig1.add_subplot(gs[0])
ax1.plot(t, z_hist[0, :], 'k.', markersize=2, alpha=0.15, label='Noisy Accel')
ax1.plot(t, x_est_hist[0, :], 'b-', linewidth=2, label='KF Estimate')
# Plot Confidence Bounds
ax1.fill_between(t, 
                 x_est_hist[0, :] - threeSigma_hist[0, :], 
                 x_est_hist[0, :] + threeSigma_hist[0, :], 
                 color='blue', alpha=0.1)
ax1.set_ylabel('Roll (deg)')
ax1.set_title('Tracking Performance: Estimate vs Measurement')
ax1.legend(loc='upper right')
ax1.grid(True)

# Plot Pitch
ax2 = fig1.add_subplot(gs[1])
ax2.plot(t, z_hist[1, :], 'k.', markersize=2, alpha=0.15, label='Noisy Accel')
ax2.plot(t, x_est_hist[1, :], 'g-', linewidth=2, label='KF Estimate')
ax2.fill_between(t, 
                 x_est_hist[1, :] - threeSigma_hist[1, :], 
                 x_est_hist[1, :] + threeSigma_hist[1, :], 
                 color='green', alpha=0.1)
ax2.set_ylabel('Pitch (deg)')
ax2.set_xlabel('Time (s)')
ax2.legend(loc='upper right')
ax2.grid(True)

plt.tight_layout()
plt.show()

# --- FIGURE 2: RESIDUAL ANALYSIS (The "Is it working?" Plot) ---
# We verify if the residuals (Innovation) are Gaussian white noise
residuals = z_hist - x_est_hist[0:2, :] # Meas - Est

fig2 = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

# Left: Residuals over Time
ax3 = fig2.add_subplot(gs[0])
ax3.plot(t, residuals[0, :], 'b', alpha=0.6, linewidth=1, label='Roll Residual')
ax3.plot(t, residuals[1, :], 'g', alpha=0.6, linewidth=1, label='Pitch Residual')
ax3.axhline(0, color='k', linestyle='--', linewidth=1)
ax3.set_title('Residuals (Measurement - Estimate)')
ax3.set_ylabel('Error (degrees)')
ax3.set_xlabel('Time (s)')
ax3.legend()
ax3.grid(True, which='both', alpha=0.3)

# Right: Histogram of Residuals
ax4 = fig2.add_subplot(gs[1])
ax4.hist(residuals[0, :], bins=50, density=True, orientation='horizontal', color='b', alpha=0.5, label='Roll')
ax4.hist(residuals[1, :], bins=50, density=True, orientation='horizontal', color='g', alpha=0.5, label='Pitch')

# Plot ideal Gaussian fit for comparison
mu, std = norm.fit(residuals[0, :])
xmin, xmax = ax4.get_ylim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, 0, std) # Ideal Gaussian centered at 0
ax4.plot(p, x, 'k--', linewidth=2, label='Ideal Gaussian')

ax4.set_title('Error Distribution')
ax4.set_xlabel('Probability Density')
ax4.legend(loc='upper right')
ax4.grid(True)

plt.tight_layout()
plt.show()


# --- FIGURE 3: BIAS CONVERGENCE & CONFIDENCE ---
# Shows if the filter "learned" the sensor defects
fig3 = plt.figure(figsize=(12, 6))

# Plot Bias
plt.plot(t, x_est_hist[2, :], 'r-', linewidth=2, label='Learned Roll Bias')
plt.plot(t, x_est_hist[3, :], 'orange', linewidth=2, linestyle='-', label='Learned Pitch Bias')

# Plot 3-Sigma Bounds for Bias (Shows filter confidence shrinking)
# We offset the bounds by the estimate to visualize them around the line
plt.fill_between(t, 
                 x_est_hist[2, :] - threeSigma_hist[2, :], 
                 x_est_hist[2, :] + threeSigma_hist[2, :], 
                 color='red', alpha=0.15)

plt.fill_between(t, 
                 x_est_hist[3, :] - threeSigma_hist[3, :], 
                 x_est_hist[3, :] + threeSigma_hist[3, :], 
                 color='orange', alpha=0.15)

plt.title('Internal State Health: Gyro Bias Learning')
plt.ylabel('Bias Magnitude (deg/s)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True)
plt.show()
