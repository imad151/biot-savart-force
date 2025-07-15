import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# === CONFIG ===
z_target = 1.0     # mm, slice location
z_tol = 0.0001     # mm, tolerance

# Load CSV
df = pd.read_csv(r"C:\Users\lab\Desktop\Personal Projects\RemoteChess\Biot-Savart\bfield_3d.csv")

# Slice data near desired Z
slice_df = df[np.abs(df["Z_mm"] - z_target) < z_tol]


if slice_df.empty:
    raise ValueError(f"No data found at Z = {z_target} ± {z_tol} mm")

# Get XY coords and field magnitude in mT
X = slice_df["X_mm"].values
Y = slice_df["Y_mm"].values
Bx = slice_df["Bx_T"].values
By = slice_df["By_T"].values
Bz = slice_df["Bz_T"].values
B_mag_mT = 1e3 * np.sqrt(Bx**2 + By**2 + Bz**2)  # Convert to mT

# Interpolate to grid
grid_x, grid_y = np.meshgrid(
    np.linspace(X.min(), X.max(), 300),
    np.linspace(Y.min(), Y.max(), 300)
)
grid_B = griddata((X, Y), B_mag_mT, (grid_x, grid_y), method='linear')

# Plot
plt.figure(figsize=(10, 8))
c = plt.pcolormesh(grid_x, grid_y, grid_B, shading='auto', cmap='plasma')
plt.colorbar(c, label='|B| (mT)')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.title(f'Magnetic Field Magnitude at Z = {z_target} mm')
plt.tight_layout()
plt.show()
