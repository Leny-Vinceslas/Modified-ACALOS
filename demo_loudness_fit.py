# demo_loudness_fit.py
# This script demonstrates fitting loudness functions using both BTUX and BTX models

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

from loudness_fit_engine import run_loudness_fit
from loudness_model_wrapper import evaluate_loudness_fixed_lcut

# Helper function to truncate data at 50 CU
def truncate_to_50_cu(x_vals, y_vals):
    #Keep y<50 
    below_mask = y_vals < 50
    x_sub = x_vals[below_mask]
    y_sub = y_vals[below_mask]

    ge_mask = y_vals >= 50
    if ge_mask.any():
        first_ge = int(np.argmax(ge_mask))  # first index with y>=50
        x_sub = np.append(x_sub, x_vals[first_ge])
        y_sub = np.append(y_sub, y_vals[first_ge])

    return x_sub, y_sub

# -----------------------------
# 1. Define Measured Data Points
# -----------------------------
np.random.seed(42)

levels = np.array([18, 22, 26, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85], dtype=float)
cu = np.array([0, 3, 4, 5, 8, 10, 12, 15, 18, 22, 26, 30, 38, 45, 50], dtype=float)

jitter_factor = 1.0 + 0.1 * (2 * np.random.rand(len(levels)) - 1)
levels_jittered = levels * jitter_factor

measured_data_jittered = np.empty(levels.size * 2)
measured_data_jittered[0::2] = levels_jittered
measured_data_jittered[1::2] = cu

# -----------------------------
# 2. Fit Using Both BTUX and BTX
# -----------------------------
params_btux = run_loudness_fit(measured_data_jittered, fit_mode="BTUX", optAlg="NEL", defaultUpperSlope=1.39)
params_btx = run_loudness_fit(measured_data_jittered, fit_mode="BTX", optAlg="NEL", defaultUpperSlope=1.39)

print("BTUX params [Lcut, m_low, m_high]:", params_btux)
print("BTX params  [Lcut, m_low, m_high]:", params_btx)

x_fit = np.linspace(levels_jittered.min(), 100, 400)
y_btux = evaluate_loudness_fixed_lcut(x_fit, params_btux)
y_btx = evaluate_loudness_fixed_lcut(x_fit, params_btx)

x_btux, y_btux = truncate_to_50_cu(x_fit, y_btux)
x_btx, y_btx = truncate_to_50_cu(x_fit, y_btx)

# -----------------------------
# 3. Plot Comparison of BTUX and BTX Fits
# -----------------------------

plt.figure(figsize=(7, 5))
plt.scatter(levels_jittered, cu, label="data points", zorder=3, color="tab:blue")
plt.plot(x_btux, y_btux,label="BTUX fit", linewidth=2, color="tab:orange")
plt.plot(x_btx, y_btx,'--', label="BTX fit", linewidth=2, color="tab:green")
plt.xlabel("Level (dB SPL)")
plt.ylabel("Categorical Units (CU)")
plt.title("Loudness Function Fits (BTUX vs BTX)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

