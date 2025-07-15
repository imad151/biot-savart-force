import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# === MAGNET CONFIG (you can tweak these) ===

# Neodymium magnet properties (typical N52 grade)
magnet = {
    "mass_g": 10.0,               # mass in grams
    "volume_cm3": 1.0,            # volume in cubic cm (for reference)
    "magnetization_A_per_m": 1.3e6,  # approx saturation magnetization (A/m)
    "magnetic_moment_A_m2": 0.01, # magnetic moment (A·m²), approx = magnetization * volume (m³)
    "dimensions_mm": (10, 10, 10)  # length, width, height (mm)
}

# Magnetic moment vector (pointing z-dir)
m = np.array([0, 0, magnet["magnetic_moment_A_m2"]])  # A·m²

# === FIELD DATA CONFIG ===
csv_file = r"C:\Users\lab\Desktop\Personal Projects\RemoteChess\Biot-Savart\bfield_3d.csv"
delta = 0.1  # mm for numerical gradient

# === PATH CONFIG ===
x_fixed = 0.0  # fixed x position (mm)
y_fixed = 0.0  # fixed y position (mm)

# === Load magnetic field data ===
df = pd.read_csv(csv_file)

# Prepare grid
x = np.sort(df["X_mm"].unique())
y = np.sort(df["Y_mm"].unique())
z = np.sort(df["Z_mm"].unique())
shape = (len(x), len(y), len(z))

Bx = df["Bx_T"].values.reshape(shape)
By = df["By_T"].values.reshape(shape)
Bz = df["Bz_T"].values.reshape(shape)

# Interpolators
Bx_interp = RegularGridInterpolator((x, y, z), Bx)
By_interp = RegularGridInterpolator((x, y, z), By)
Bz_interp = RegularGridInterpolator((x, y, z), Bz)

# Define path along Z (avoid boundaries)
z_path = np.linspace(z.min() + delta, z.max() - delta, 200)
positions = np.column_stack([np.full_like(z_path, x_fixed),
                             np.full_like(z_path, y_fixed),
                             z_path])

# Compute force on magnet at each position
forces = []
for pos in positions:
    F = np.zeros(3)
    for i, B_interp in enumerate([Bx_interp, By_interp, Bz_interp]):
        grad = np.zeros(3)
        for j in range(3):
            dp = np.zeros(3)
            dp[j] = delta
            B_plus = B_interp(pos + dp)
            B_minus = B_interp(pos - dp)
            grad[j] = (B_plus - B_minus) / (2 * delta)
        F[i] = np.dot(m, grad)
    forces.append(F)

forces = np.array(forces)
force_mags = np.linalg.norm(forces, axis=1)

# === Plot Force Magnitude vs Distance ===
plt.figure(figsize=(10, 6))
plt.plot(z_path, force_mags, label='|Force| (N)', color='crimson')
plt.xlabel('Z Position (mm)')
plt.ylabel('Force (N)')
plt.title('Force on Neodymium Magnet Along Z-axis')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === BONUS: Print summary ===
print(f"Magnet Mass: {magnet['mass_g']} g")
print(f"Magnetic Moment: {magnet['magnetic_moment_A_m2']} A·m²")
print(f"Force range: {force_mags.min():.3e} N to {force_mags.max():.3e} N")
