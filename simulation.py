import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import Kalmanfilter as kf
import mass_spring_dampersystem as msds 

msds = msds.MassSpringDamperSystem(mass=1.0, spring_constant=2.0, damping_coefficient=0.5)
x_hist, z_hist, t_hist = msds.simulation()

init_kalman = kf.KalmanFilter(
    A=msds.Ad,
    B=msds.Bd,
    C=msds.Cd,
    Q=msds.Qd,
    R=msds.Rd,
    P=msds.Pd,
    x0=msds.x0d
)
num_steps = x_hist.shape[1]
x_est_hist = np.zeros((2, num_steps))  
x_est_hist[:, 0:1] = msds.x0d
for k in range(num_steps - 1):
    u_k = msds.F[:, k:k+1]  
    z_k = z_hist[:, k:k+1]  
    
    init_kalman.predict(u_k)
    x_est = init_kalman.update(z_k)
    
    x_est_hist[:, k+1:k+2] = x_est
# Plotting Results
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t_hist, x_hist[0, :], label='True Position', color='blue')
plt.plot(t_hist, z_hist[0, :], 'r.', markersize=2, alpha=0.3, label='Noisy Measurement')
plt.plot(t_hist, x_est_hist[0, :], label='Kalman Filter Estimate', color='green')
plt.title('Kalman Filter State Estimation')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(t_hist, x_hist[1, :], label='True Velocity', color='blue')
plt.plot(t_hist, x_est_hist[1, :], label='Kalman Filter Estimate', color='green')
plt.ylabel('Velocity (m/s)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Plotting Results
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t_hist, x_hist[0, :], label='True Position', color='blue')
plt.plot(t_hist, z_hist[0, :], 'r.', markersize=2, alpha=0.3, label='Noisy Measurement')
plt.plot(t_hist, x_est_hist[0, :], label='Kalman Filter Estimate', color='green')
plt.title('Kalman Filter State Estimation')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(t_hist, x_hist[1, :], label='True Velocity', color='blue')
plt.plot(t_hist, x_est_hist[1, :], label='Kalman Filter Estimate', color='green')
plt.ylabel('Velocity (m/s)')   
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
