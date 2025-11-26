#%%
import numpy as np
import matplotlib.pyplot as plt

from fit_loudness_function import fit_loudness_function
from loudness_function import loudness_function
from loudness_function import loudness_function_Lcut
#%%

# -----------------------------
# 1. Original Estimated Datapoints
# -----------------------------

levels = np.array([18, 22, 26, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85], dtype=float)
cu     = np.array([ 0,  3,  4,  5,  8, 10, 12, 15, 18, 22, 26, 30, 38, 45, 50], dtype=float)

measured_data = np.empty(levels.size * 2)
measured_data[0::2] = levels
measured_data[1::2] = cu


# -----------------------------
# 2. Add ±10% Random Jitter on X-Axis
# -----------------------------

np.random.seed(42)

jitter_factor = 1.0 + 0.1 * (2 * np.random.rand(len(levels)) - 1)
levels_jittered = levels * jitter_factor

measured_data_jittered = np.empty(levels.size * 2)
measured_data_jittered[0::2] = levels_jittered
measured_data_jittered[1::2] = cu


# -----------------------------
# 3. Fit Using BTUX
# -----------------------------
#%%
fit_method = 'BTUX'
min_method = 'NEL'
defaultUpperSlopefit = 1.39

fit_params = fit_loudness_function(
    measured_data_jittered,
    fit_mode=fit_method,
    optAlg=min_method,
    defaultUpperSlope=defaultUpperSlopefit
)

print("Fitted parameters [Lcut, m_low, m_high]:")
print(fit_params)


# -----------------------------
# 4. Generate Fitted Loudness Curve
# -----------------------------


x_fit = np.linspace(levels_jittered.min(), 100, 400)
y_fit = loudness_function_Lcut(x_fit, fit_params)

#keep the data points below 50 CU from x_fit and y_fit and 
# and only keep the first data point where y_fit reaches 50 CU.
mask = y_fit <= 50
x_fit = x_fit[mask]
y_fit = y_fit[mask]
if np.any(y_fit == 50):
    first_50_index = np.where(y_fit == 50)[0][0]
    x_fit = x_fit[:first_50_index + 1]
    y_fit = y_fit[:first_50_index + 1]


#%%
# -----------------------------
# 5. Plot Jittered Data & Fitted Curve
# -----------------------------

plt.figure(figsize=(7, 5))

plt.scatter(levels_jittered, cu, label="data points", zorder=3)
plt.plot(x_fit, y_fit, label="BTUX fit (NEL, m_high = 1.39)", linewidth=2)
plt.xlabel("Level (dB SPL)")
plt.ylabel("Categorical Units (CU)")
plt.title("BTUX Loudness Function Fit with ±10% X-Jitter")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

