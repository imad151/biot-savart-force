import warnings ; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import magpylib as mag
import pandas as pd
from scipy.interpolate import splprep, splev
import csv
import gc

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

def save_3d_field_chunked(current_line, x_range_mm, y_range_mm, z_range_mm, xy_res, z_res, chunk_size=1000):
    """
    Save 3D B-field data in chunks to avoid memory issues.
    Processes data in smaller batches while maintaining full resolution.
    """
    x = y = np.linspace(x_range_mm[0], x_range_mm[1], xy_res)
    X, Y = np.meshgrid(x, y)
    
    z_values = np.arange(z_range_mm[0], z_range_mm[1] + z_res, z_res)
    total_points = xy_res * xy_res * len(z_values)
    
    print(f"Total points to calculate: {total_points:,}")
    print(f"Processing in chunks of {chunk_size:,} points")
    
    with open("bfield_3d.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["X_mm", "Y_mm", "Z_mm", "Bx_T", "By_T", "Bz_T"])
        
        processed_points = 0
        
        for z_mm in z_values:
            print(f"Processing slice at Z = {z_mm:.2f} mm")
            Z = np.full_like(X, z_mm)
            pos_3d_mm = np.stack([X, Y, Z], axis=-1)
            pos_3d_m = pos_3d_mm.reshape(-1, 3) / 1000.0  # meters
            
            # Process this slice in chunks
            slice_points = pos_3d_m.shape[0]
            for start_idx in range(0, slice_points, chunk_size):
                end_idx = min(start_idx + chunk_size, slice_points)
                chunk_pos_m = pos_3d_m[start_idx:end_idx]
                chunk_pos_mm = pos_3d_mm.reshape(-1, 3)[start_idx:end_idx]
                
                # Calculate B-field for this chunk
                B_chunk = current_line.getB(chunk_pos_m)
                
                # Write chunk data
                for i in range(B_chunk.shape[0]):
                    writer.writerow([
                        chunk_pos_mm[i, 0],
                        chunk_pos_mm[i, 1], 
                        chunk_pos_mm[i, 2],
                        B_chunk[i, 0], B_chunk[i, 1], B_chunk[i, 2]
                    ])
                
                processed_points += (end_idx - start_idx)
                progress = (processed_points / total_points) * 100
                print(f"  Progress: {progress:.1f}% ({processed_points:,}/{total_points:,})")
                
                # Force garbage collection to free memory
                del B_chunk
                gc.collect()
    
    print(f"Completed! Saved {processed_points:,} points to bfield_3d.csv")

def save_3d_field_slice_by_slice_optimized(current_line, x_range_mm, y_range_mm, z_range_mm, xy_res, z_res):
    """
    Optimized version that processes one Z-slice at a time and uses efficient memory management.
    """
    x = y = np.linspace(x_range_mm[0], x_range_mm[1], xy_res)
    X, Y = np.meshgrid(x, y)
    
    z_values = np.arange(z_range_mm[0], z_range_mm[1] + z_res, z_res)
    total_slices = len(z_values)
    points_per_slice = xy_res * xy_res
    
    print(f"Processing {total_slices} slices with {points_per_slice:,} points each")
    print(f"Total points: {total_slices * points_per_slice:,}")
    
    with open("bfield_3d.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["X_mm", "Y_mm", "Z_mm", "Bx_T", "By_T", "Bz_T"])
        
        for slice_idx, z_mm in enumerate(z_values):
            print(f"Slice {slice_idx + 1}/{total_slices}: Z = {z_mm:.2f} mm")
            
            Z = np.full_like(X, z_mm)
            pos_3d_mm = np.stack([X, Y, Z], axis=-1)
            pos_3d_m = pos_3d_mm.reshape(-1, 3) / 1000.0  # meters
            
            # Calculate B-field for entire slice at once (should be manageable)
            B = current_line.getB(pos_3d_m)
            
            # Write slice data efficiently
            pos_flat_mm = pos_3d_mm.reshape(-1, 3)
            for i in range(B.shape[0]):
                writer.writerow([
                    pos_flat_mm[i, 0], pos_flat_mm[i, 1], pos_flat_mm[i, 2],
                    B[i, 0], B[i, 1], B[i, 2]
                ])
            
            # Clean up memory
            del B, pos_3d_mm, pos_3d_m, Z, pos_flat_mm
            gc.collect()
            
            progress = ((slice_idx + 1) / total_slices) * 100
            print(f"  Progress: {progress:.1f}%")

def calculate_field_statistics(current_line, x_range_mm, y_range_mm, z_range_mm, xy_res, z_res):
    """
    Calculate and display statistics about the B-field without storing all data.
    Useful for understanding field characteristics before full calculation.
    """
    x = y = np.linspace(x_range_mm[0], x_range_mm[1], xy_res)
    X, Y = np.meshgrid(x, y)
    
    # Sample a few representative slices
    z_sample = np.linspace(z_range_mm[0], z_range_mm[1], 5)
    
    b_max_overall = 0
    b_min_overall = float('inf')
    
    print("Calculating field statistics on sample slices...")
    
    for z_mm in z_sample:
        Z = np.full_like(X, z_mm)
        pos_3d_mm = np.stack([X, Y, Z], axis=-1)
        pos_3d_m = pos_3d_mm.reshape(-1, 3) / 1000.0
        
        B = current_line.getB(pos_3d_m)
        B_mag = np.linalg.norm(B, axis=1)
        
        b_max_slice = np.max(B_mag)
        b_min_slice = np.min(B_mag)
        b_mean_slice = np.mean(B_mag)
        
        print(f"Z = {z_mm:.1f} mm: |B| range = [{b_min_slice:.2e}, {b_max_slice:.2e}] T, mean = {b_mean_slice:.2e} T")
        
        b_max_overall = max(b_max_overall, b_max_slice)
        b_min_overall = min(b_min_overall, b_min_slice)
    
    print(f"\nOverall |B| range (sample): [{b_min_overall:.2e}, {b_max_overall:.2e}] T")
    return b_min_overall, b_max_overall

if __name__ == "__main__":
    # Generate coil geometry
    layers = generate_stacked_coils(
        copper_width=0.15,
        trace_spacing=0.15,
        num_turns=14,
        outer_diameter=9.7,
        num_layers=4,
        layer_height=0.2
    )

    # Resample coil for smoothness and computational efficiency
    layers_resampled = resample_vertices(layers, num_points=2000)
    current_line = mag.current.Line(current=1.0, vertices=layers_resampled)

    # Grid parameters
    plane_size_mm = 43.18
    resolution_xy = 100  # 100x100 grid in XY plane
    z_min, z_max, z_step = 0, 10, 0.1  # Z range with 0.1 mm steps

    # Save full 3D field data using optimized slice-by-slice approach
    print("\n=== Calculating Full 3D Field ===")
    save_3d_field_slice_by_slice_optimized(
        current_line, 
        x_range_mm=[-plane_size_mm/2, plane_size_mm/2], 
        y_range_mm=[-plane_size_mm/2, plane_size_mm/2], 
        z_range_mm=[z_min, z_max], 
        xy_res=resolution_xy,
        z_res=z_step
    )

    # Plot selected planar slices for visualization
    print("\n=== Generating Visualization ===")
    plot_field_slices(layers_resampled, z_slices_mm=[0, 5, 10], plane_size_mm=plane_size_mm, resolution_xy=resolution_xy)
    
    print("\nProcessing complete!")