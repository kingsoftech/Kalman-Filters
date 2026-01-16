Here is the updated `README.md` with the system diagram inserted right in the center, between the mathematical model and the project structure.

**Note:** Save the image I generated for you in the previous step as `system_diagram.png` and place it in your project folder.

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

### Continuous System

The system is defined by the differential equation:


Converted to State-Space form :

### Discrete Kalman Filter

The continuous system is discretized with time step :

1. **State Transition:** 
2. **Process Noise ():** Computed via the matrix exponential of the Hamiltonian block matrix (Van Loan's Method).
3. **Predict Step:**
*   $\hat{x}_{k|k-1} = A_d \hat{x}_{k-1|k-1} + B_d u_k
    P_{k|k-1} = A_d P_{k-1|k-1} A_d^T + Q_d$
* 


4. **Update Step:**
* 
* 
* 



---

## 🖼️ System Diagram

The following schematic illustrates the physical parameters () simulated in this project:

*Figure 3: Schematic of the Mass-Spring-Damper System*

---

## 📂 Project Structure

```
.
├── simulation.py       # Main script containing System and Filter classes
├── Figure_1.png        # Plot of State Estimates vs Ground Truth
├── Figure_2.png        # Plot of Estimation Errors with 3-Sigma bounds
├── system_diagram.png  # Schematic of the mechanical system
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

## 📜 License

This project is open-source and available under the MIT License.