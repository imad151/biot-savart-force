import numpy as np
import matplotlib.pyplot as plt
import magpylib as mag

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


# === Field Plotter ===
def plot_field_slices(coil_path, z_slices_mm, plane_size_mm=10, resolution=100):
    # Create Magpylib current line (1 A)
    current_line = mag.current.Line(current=1.0, vertices=coil_path)

    # Create XY grid (in meters)
    x = y = np.linspace(-plane_size_mm/2, plane_size_mm/2, resolution)
    X, Y = np.meshgrid(x, y)
    positions_mm = np.stack([X, Y], axis=-1)

    fig, axs = plt.subplots(1, len(z_slices_mm), figsize=(5 * len(z_slices_mm), 5))
    if len(z_slices_mm) == 1:
        axs = [axs]

    for ax, z_mm in zip(axs, z_slices_mm):
        Z = np.full_like(X, z_mm)
        pos_3d_mm = np.stack([X, Y, Z], axis=-1)
        pos_3d_m = pos_3d_mm / 1000.0  # convert to meters

        B = current_line.getB(pos_3d_m)  # [T]
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

# === Run it ===
if __name__ == "__main__":
    layers = generate_stacked_coils(
        copper_width=0.15,
        trace_spacing=0.15,
        num_turns=14,
        outer_diameter=9.7,
        num_layers=4,
        layer_height=0.2
    )

    z_slices = [0, 2.5, 5.0]  # mm
    plot_field_slices(layers, z_slices_mm=z_slices, plane_size_mm=10, resolution=200)
