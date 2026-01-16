Here is a professional, ready-to-use `README.md` for your project. You can copy and paste this directly into your GitHub repository or project folder.

---

# Kalman Filter for Mass-Spring-Damper System

A Python implementation of a discrete-time Kalman Filter used to estimate the state (position and velocity) of a Mass-Spring-Damper system. This project demonstrates advanced control theory concepts, including **Van Loan’s method** for discretizing process noise covariance.

*Figure 1: Comparison of True State vs. Kalman Filter Estimate*

## 📝 Overview

This project simulates a mechanical Mass-Spring-Damper system driven by a sinusoidal force. It generates noisy measurements of the position and uses a Kalman Filter to reconstruct the true state vector (position and velocity) from these noisy observations.

**Key Features:**

* **Physics Simulation:** Accurate state-space modeling of a 2nd-order mechanical system.
* **Advanced Discretization:** Uses the Matrix Exponential (`scipy.linalg.expm`) for the State Transition Matrix ().
* **Van Loan’s Method:** Implements the robust Van Loan method to compute the discrete Process Noise Covariance matrix (), ensuring numerical stability and physical accuracy.
* **Performance Analysis:** Includes visualization of  (3-sigma) error bounds to validate filter consistency.

## ⚙️ Mathematical Model
![System Schematic](massspringdamper.png)
### Continuous System

The system is defined by the differential equation:


Converted to State-Space form :

### Discrete Kalman Filter

The continuous system is discretized with time step :

1. **State Transition:** 
2. **Process Noise ():** Computed via the matrix exponential of the Hamiltonian block matrix (Van Loan's Method).
3. **Predict Step:**
* 
* 


4. **Update Step:**
* 
* 
* 



## 📂 Project Structure

```
.
├── simulation.py       # Implements script containing System and Filter classes
├── Figure_1.png        # Plot of State Estimates vs Ground Truth
├── Figure_2.png        # Plot of Estimation Errors with 3-Sigma bounds
└── README.md           # Project documentation

```

## 🚀 Getting Started

### Prerequisites

You will need Python 3.x and the following libraries:

```bash
pip install numpy scipy matplotlib

```

### Running the Simulation

1. Clone the repository or download the script.
2. Run the main Python script:
```bash
python simulation.py

```


3. The script will generate two plots:
* **Figure 1:** Shows the true position/velocity overlaid with the noisy measurements and the Kalman Filter estimate.
* **Figure 2:** Shows the error residuals (True - Estimate) bounded by the  confidence interval.



## 📊 Results

The filter successfully tracks the sinusoidal motion despite significant measurement noise.

**Error Analysis:**
As seen in `Figure_2.png`, the estimation error (blue line) stays consistently within the  bounds (shaded region). This indicates the filter is consistent and the noise matrices ( and ) are tuned correctly.

*Figure 2: Estimation Error with  Confidence Bounds*

## 🛠 Configuration

You can modify the system parameters in the `main` block of the script to test different scenarios:

```python
# System Parameters
mass = 50.0              # Mass (kg)
spring_constant = 2.0    # Stiffness (N/m)
damping_coefficient = 4.0 # Damping (N-s/m)
dt = 0.1                 # Time step (s)

# Noise Parameters (Inside models method)
Sw = 0.01                # Process noise spectral density
Sv = 0.00001             # Measurement noise standard deviation

```
