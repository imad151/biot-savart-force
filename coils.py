import warnings ; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import magpylib as mag
import pandas as pd
from scipy.interpolate import splprep, splev

# === Spiral Generator (in mm) ===
def generate_spiral_coil(
    copper_width=0.15,
    trace_spacing=0.15,
    num_turns=14,
    outer_diameter=9.7,
    num_points_per_turn=200,
    layer_height=0,
    layer_index=0
):
    total_spacing = copper_width + trace_spacing
    inner_diameter = outer_diameter - 2 * total_spacing * num_turns
    if inner_diameter <= 0:
        raise ValueError("Invalid geometry: inner_diameter <= 0")

    theta = np.linspace(0, 2 * np.pi * num_turns, num_turns * num_points_per_turn)
    r = np.linspace(inner_diameter / 2, outer_diameter / 2, len(theta))

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.full_like(x, layer_height * layer_index)  # vertical offset

    # Convert mm → meters for magpylib
    return np.vstack((x, y, z)).T / 1000.0

def generate_stacked_coils(**kwargs):
    num_layers = kwargs.pop("num_layers")
    layers = []
    for i in range(num_layers):
        layer = generate_spiral_coil(layer_index=i, **kwargs)
        layers.append(layer)
    return np.vstack(layers)

def resample_vertices(vertices, num_points=1000):
    """Smooth and resample 3D path using spline interpolation."""
    tck, _ = splprep(vertices.T, s=0)
    u_fine = np.linspace(0, 1, num_points)
    smoothed = np.array(splev(u_fine, tck)).T
    return smoothed

def save_field_to_csv(X, Y, Z, B, filename="bfield_data.csv"):
    # Flatten all arrays
    data = np.column_stack((X.ravel(), Y.ravel(), Z.ravel(), B[:,0].ravel(), B[:,1].ravel(), B[:,2].ravel()))
    df = pd.DataFrame(data, columns=["X_mm", "Y_mm", "Z_mm", "Bx_T", "By_T", "Bz_T"])
    df.to_csv(filename, index=False)
    print(f"Saved B-field data to {filename}")

def plot_field_slices(coil_path, z_slices_mm, plane_size_mm=43.18, resolution_xy=100):
    coil_path = resample_vertices(coil_path, num_points=2000)
    current_line = mag.current.Line(current=1.0, vertices=coil_path)

    # Grid for X,Y in meters
    x = y = np.linspace(-plane_size_mm/2, plane_size_mm/2, resolution_xy)
    X, Y = np.meshgrid(x, y)

    fig, axs = plt.subplots(1, len(z_slices_mm), figsize=(5 * len(z_slices_mm), 5))
    if len(z_slices_mm) == 1:
        axs = [axs]

    for ax, z_mm in zip(axs, z_slices_mm):
        Z = np.full_like(X, z_mm)
        pos_3d_mm = np.stack([X, Y, Z], axis=-1)
        pos_3d_m = pos_3d_mm.reshape(-1, 3) / 1000.0

        B = current_line.getB(pos_3d_m).reshape(X.shape + (3,))
        B_mag = np.linalg.norm(B, axis=-1)

        im = ax.imshow(
            B_mag,
            extent=[-plane_size_mm/2, plane_size_mm/2, -plane_size_mm/2, plane_size_mm/2],
            origin='lower',
            cmap='plasma'
        )
        ax.set_title(f"B field at z = {z_mm:.1f} mm")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        fig.colorbar(im, ax=ax, label="|B| [T]")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    layers = generate_stacked_coils(
        copper_width=0.15,
        trace_spacing=0.15,
        num_turns=14,
        outer_diameter=9.7,
        num_layers=4,
        layer_height=0.2
    )

    # Fine 3D grid (mm)
    plane_size_mm = 43.18
    resolution_xy = 100  # 100x100 grid in XY plane → 10k points per slice

    z_mm = np.arange(0, 10.01, 0.1)  # from 0 to 10 mm with 0.1 mm steps → 101 slices

    # Make full 3D grid in mm
    x = y = np.linspace(-plane_size_mm/2, plane_size_mm/2, resolution_xy)
    X, Y, Z = np.meshgrid(x, y, z_mm, indexing='ij')
    pos_3d_mm = np.stack([X, Y, Z], axis=-1)
    pos_3d_m = pos_3d_mm.reshape(-1, 3) / 1000.0  # meters

    # Resample coil for smoothness and speed
    layers_resampled = resample_vertices(layers, num_points=2000)

    current_line = mag.current.Line(current=1.0, vertices=layers_resampled)

    print("Calculating B-field on full 3D grid, this may take a while...")
    B = current_line.getB(pos_3d_m)
    B = B.reshape((resolution_xy, resolution_xy, len(z_mm), 3))

    # Save CSV (flatten grid to list of points)
    save_field_to_csv(X, Y, Z, B)

    # Plot selected planar slices
    plot_field_slices(layers_resampled, z_slices_mm=[0, 10, 1], plane_size_mm=plane_size_mm, resolution=resolution_xy)
