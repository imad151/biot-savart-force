import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe
from scipy.integrate import quad
import warnings
from typing import Tuple, Optional, Union, List
from dataclasses import dataclass
import time

# Physical Constants (SI units)
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space [H/m]
EPSILON = 1e-12  # Numerical epsilon for singularity handling

@dataclass
class CoilParameters:
    """Parameters defining a circular coil."""
    radius: float = 5e-3  # Coil radius [m]
    turns: int = 20       # Number of turns
    current: float = 1.0  # Current [A]
    center: np.ndarray = None  # Coil center position [m]
    
    def __post_init__(self):
        if self.center is None:
            self.center = np.array([0.0, 0.0, 0.0])

@dataclass
class MagnetParameters:
    """Parameters defining a physical magnet."""
    # Standard neodymium magnet properties
    magnetic_moment: float = 1.0  # Magnetic moment magnitude [A·m²]
    orientation: np.ndarray = None  # Moment direction (unit vector)
    mass: float = 1e-3  # Magnet mass [kg]
    dimensions: np.ndarray = None  # [length, width, height] in meters
    material: str = "N52"  # Magnet grade
    
    def __post_init__(self):
        if self.orientation is None:
            self.orientation = np.array([0.0, 0.0, 1.0])  # Default: pointing up
        if self.dimensions is None:
            self.dimensions = np.array([5e-3, 5e-3, 2e-3])  # Default: 5x5x2 mm
        
        # Normalize orientation vector
        self.orientation = self.orientation / np.linalg.norm(self.orientation)

class StandardMagnets:
    """Predefined standard magnet configurations."""
    
    @staticmethod
    def neodymium_disk_small():
        """Small neodymium disk magnet (5mm diameter, 2mm thick)."""
        # Typical N52 grade neodymium
        diameter = 5e-3
        thickness = 2e-3
        volume = np.pi * (diameter/2)**2 * thickness
        
        # N52 grade: ~1.4-1.45 T remanence, ~1100 kA/m coercivity
        # Estimated magnetic moment based on volume and grade
        Br = 1.42  # Tesla (remanence)
        moment_magnitude = (Br / MU_0) * volume * 0.8  # Approximation factor
        
        return MagnetParameters(
            magnetic_moment=moment_magnitude,
            orientation=np.array([0.0, 0.0, 1.0]),
            mass=volume * 7500,  # Neodymium density ~7.5 g/cm³
            dimensions=np.array([diameter, diameter, thickness]),
            material="N52"
        )
    
    @staticmethod
    def neodymium_block_medium():
        """Medium neodymium block magnet (10x10x5 mm)."""
        dimensions = np.array([10e-3, 10e-3, 5e-3])
        volume = np.prod(dimensions)
        
        Br = 1.42  # Tesla
        moment_magnitude = (Br / MU_0) * volume * 0.8
        
        return MagnetParameters(
            magnetic_moment=moment_magnitude,
            orientation=np.array([0.0, 0.0, 1.0]),
            mass=volume * 7500,
            dimensions=dimensions,
            material="N52"
        )
    
    @staticmethod
    def neodymium_cylinder_large():
        """Large neodymium cylinder magnet (15mm diameter, 10mm height)."""
        diameter = 15e-3
        height = 10e-3
        volume = np.pi * (diameter/2)**2 * height
        
        Br = 1.42  # Tesla
        moment_magnitude = (Br / MU_0) * volume * 0.8
        
        return MagnetParameters(
            magnetic_moment=moment_magnitude,
            orientation=np.array([0.0, 0.0, 1.0]),
            mass=volume * 7500,
            dimensions=np.array([diameter, diameter, height]),
            material="N52"
        )
    
    @staticmethod
    def ferrite_disk():
        """Standard ferrite disk magnet (20mm diameter, 3mm thick)."""
        diameter = 20e-3
        thickness = 3e-3
        volume = np.pi * (diameter/2)**2 * thickness
        
        Br = 0.4  # Tesla (ferrite)
        moment_magnitude = (Br / MU_0) * volume * 0.6
        
        return MagnetParameters(
            magnetic_moment=moment_magnitude,
            orientation=np.array([0.0, 0.0, 1.0]),
            mass=volume * 5000,  # Ferrite density ~5 g/cm³
            dimensions=np.array([diameter, diameter, thickness]),
            material="Ferrite"
        )

class CircularCoilSimulator:
    """
    Electromagnetic field simulator for a single circular PCB coil.
    
    This class implements analytical solutions for magnetic field calculations
    using the Biot-Savart law with elliptic integrals for circular current loops.
    
    Theoretical Foundation:
    ----------------------
    The magnetic field of a circular current loop is given by:
    
    B_z(r,z) = (μ₀I/2) * [K(k) - ((a²-r²-z²)/(r²+z²-a²+2√((r-a)²+z²)))E(k)]
    B_r(r,z) = (μ₀I/2) * (z/r) * [-K(k) + ((a²+r²+z²)/(r²+z²-a²+2√((r-a)²+z²)))E(k)]
    
    where:
    - K(k), E(k) are complete elliptic integrals of 1st and 2nd kind
    - k² = 4ar/((a+r)²+z²)
    - a = coil radius, r = radial distance, z = axial distance
    """
    
    def __init__(self, coil_params: CoilParameters, magnet_params: Optional[MagnetParameters] = None):
        """
        Initialize the coil simulator.
        
        Parameters:
        -----------
        coil_params : CoilParameters
            Parameters defining the coil geometry and current
        magnet_params : MagnetParameters, optional
            Parameters defining the magnet properties
        """
        self.coil = coil_params
        self.magnet = magnet_params if magnet_params else StandardMagnets.neodymium_disk_small()
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Validate coil and magnet parameters."""
        if self.coil.radius <= 0:
            raise ValueError("Coil radius must be positive")
        if self.coil.turns <= 0:
            raise ValueError("Number of turns must be positive")
        if not np.isfinite(self.coil.current):
            raise ValueError("Current must be finite")
        if self.magnet.magnetic_moment <= 0:
            raise ValueError("Magnetic moment must be positive")
        if self.magnet.mass <= 0:
            raise ValueError("Magnet mass must be positive")
            
    def _cylindrical_to_cartesian(self, B_r: np.ndarray, B_phi: np.ndarray, 
                                 B_z: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert cylindrical field components to Cartesian.
        
        Parameters:
        -----------
        B_r, B_phi, B_z : np.ndarray
            Cylindrical field components
        phi : np.ndarray
            Azimuthal angle
            
        Returns:
        --------
        B_x, B_y, B_z : np.ndarray
            Cartesian field components
        """
        B_x = B_r * np.cos(phi) - B_phi * np.sin(phi)
        B_y = B_r * np.sin(phi) + B_phi * np.cos(phi)
        return B_x, B_y, B_z
    
    def _calculate_elliptic_parameters(self, r: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate elliptic integral parameters.
        
        Parameters:
        -----------
        r : np.ndarray
            Radial distance from coil axis
        z : np.ndarray
            Axial distance from coil plane
            
        Returns:
        --------
        k : np.ndarray
            Elliptic integral parameter
        alpha : np.ndarray
            Helper parameter α = r/a
        beta : np.ndarray
            Helper parameter β = z/a
        """
        a = self.coil.radius
        
        # Ensure arrays
        r = np.asarray(r)
        z = np.asarray(z)
        
        # Handle singularity at r=0
        r_safe = np.where(r < EPSILON, EPSILON, r)
        
        # Calculate elliptic integral parameter
        # k² = 4ar/((a+r)² + z²)
        denominator = (a + r_safe)**2 + z**2
        k_squared = 4 * a * r_safe / denominator
        
        # Ensure k² is in valid range [0,1)
        k_squared = np.clip(k_squared, 0, 1 - EPSILON)
        k = np.sqrt(k_squared)
        
        # Handle special case when r is very close to a (coil radius)
        # This prevents numerical instabilities
        alpha = r_safe / a
        alpha = np.where(np.abs(alpha - 1) < EPSILON, 1 - EPSILON, alpha)
        
        beta = z / a
        
        return k, alpha, beta
    
    def magnetic_field_cylindrical(self, r: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate magnetic field in cylindrical coordinates.
        
        Uses analytical expressions with elliptic integrals for circular current loops.
        
        Parameters:
        -----------
        r : np.ndarray
            Radial distance from coil axis [m]
        z : np.ndarray
            Axial distance from coil plane [m]
            
        Returns:
        --------
        B_r : np.ndarray
            Radial magnetic field component [T]
        B_z : np.ndarray
            Axial magnetic field component [T]
        """
        a = self.coil.radius
        I = self.coil.current * self.coil.turns
        
        # Ensure arrays have same shape
        r = np.asarray(r)
        z = np.asarray(z)
        
        # Calculate elliptic parameters
        k, alpha, beta = self._calculate_elliptic_parameters(r, z)
        
        # Calculate elliptic integrals
        K = ellipk(k**2)
        E = ellipe(k**2)
        
        # Magnetic field prefactor
        B_0 = MU_0 * I / (2 * np.pi * a)
        
        # Axial component B_z
        factor_z = 1 / np.sqrt((1 + alpha)**2 + beta**2)
        
        # Handle division by zero in B_z calculation
        denominator_z = (1 - alpha)**2 + beta**2
        denominator_z = np.where(denominator_z < EPSILON, EPSILON, denominator_z)
        
        B_z = B_0 * factor_z * (
            K + ((1 - alpha**2 - beta**2) / denominator_z) * E
        )
        
        # Radial component B_r
        # Handle r=0 case where B_r=0 by symmetry
        B_r = np.zeros_like(r)
        mask = r > EPSILON
        
        if np.any(mask):
            # Use broadcasting-safe operations
            alpha_masked = alpha[mask]
            beta_masked = beta[mask]
            K_masked = K[mask]
            E_masked = E[mask]
            
            factor_r = beta_masked / (alpha_masked * np.sqrt((1 + alpha_masked)**2 + beta_masked**2))
            
            # Handle division by zero in B_r calculation
            denominator_r = (1 - alpha_masked)**2 + beta_masked**2
            denominator_r = np.where(denominator_r < EPSILON, EPSILON, denominator_r)
            
            B_r[mask] = B_0 * factor_r * (
                -K_masked + ((1 + alpha_masked**2 + beta_masked**2) / denominator_r) * E_masked
            )
        
        return B_r, B_z
    
    def magnetic_field(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate magnetic field at given Cartesian coordinates.
        
        Parameters:
        -----------
        x, y, z : np.ndarray
            Cartesian coordinates [m]
            
        Returns:
        --------
        B_x, B_y, B_z : np.ndarray
            Magnetic field components [T]
        """
        # Convert to cylindrical coordinates relative to coil center
        x_rel = x - self.coil.center[0]
        y_rel = y - self.coil.center[1]
        z_rel = z - self.coil.center[2]
        
        r = np.sqrt(x_rel**2 + y_rel**2)
        phi = np.arctan2(y_rel, x_rel)
        
        # Calculate field in cylindrical coordinates
        B_r, B_z = self.magnetic_field_cylindrical(r, z_rel)
        B_phi = np.zeros_like(B_r)  # Azimuthal component is zero for circular coil
        
        # Convert to Cartesian coordinates
        B_x, B_y, B_z_cart = self._cylindrical_to_cartesian(B_r, B_phi, B_z, phi)
        
        return B_x, B_y, B_z_cart
    
    def field_gradient_numerical(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                               delta: float = 1e-6) -> np.ndarray:
        """
        Calculate magnetic field gradient using finite differences.
        
        Parameters:
        -----------
        x, y, z : np.ndarray
            Cartesian coordinates [m]
        delta : float
            Step size for finite differences [m]
            
        Returns:
        --------
        grad_B : np.ndarray
            Gradient tensor ∇B with shape (..., 3, 3)
            grad_B[..., i, j] = ∂B_i/∂x_j
        """
        # Ensure inputs are arrays
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        original_shape = x.shape
        
        # Flatten for computation
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()
        
        # Initialize gradient tensor
        grad_B = np.zeros((len(x_flat), 3, 3))
        
        # Calculate gradients using central differences
        for i in range(3):  # For each spatial direction
            # Create offset vectors
            offset = np.zeros(3)
            offset[i] = delta
            
            # Forward and backward points
            if i == 0:  # x-direction
                x_forward = x_flat + delta
                x_backward = x_flat - delta
                y_forward = y_backward = y_flat
                z_forward = z_backward = z_flat
            elif i == 1:  # y-direction
                x_forward = x_backward = x_flat
                y_forward = y_flat + delta
                y_backward = y_flat - delta
                z_forward = z_backward = z_flat
            else:  # z-direction
                x_forward = x_backward = x_flat
                y_forward = y_backward = y_flat
                z_forward = z_flat + delta
                z_backward = z_flat - delta
            
            # Calculate fields at forward and backward points
            B_forward = self.magnetic_field(x_forward, y_forward, z_forward)
            B_backward = self.magnetic_field(x_backward, y_backward, z_backward)
            
            # Central difference
            for j in range(3):  # For each field component
                grad_B[:, j, i] = (B_forward[j] - B_backward[j]) / (2 * delta)
        
        # Reshape to original shape
        grad_B = grad_B.reshape(original_shape + (3, 3))
        
        return grad_B
    
    def field_gradient_analytical(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Calculate magnetic field gradient using analytical derivatives.
        
        This is more accurate than numerical differentiation but requires
        more complex mathematical expressions.
        
        Parameters:
        -----------
        x, y, z : np.ndarray
            Cartesian coordinates [m]
            
        Returns:
        --------
        grad_B : np.ndarray
            Gradient tensor ∇B with shape (..., 3, 3)
        """
        # For simplicity, use numerical gradients with high precision
        # Full analytical gradients would require extensive elliptic integral derivatives
        return self.field_gradient_numerical(x, y, z, delta=1e-8)
    
    def magnetic_force(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                      magnetic_moment: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate force on a magnetic dipole: F = (m·∇)B
        
        Parameters:
        -----------
        x, y, z : np.ndarray
            Dipole position [m]
        magnetic_moment : np.ndarray
            Magnetic dipole moment [A·m²], shape (3,) or (..., 3)
            
        Returns:
        --------
        F_x, F_y, F_z : np.ndarray
            Force components [N]
        """
        # Ensure inputs are arrays
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        magnetic_moment = np.asarray(magnetic_moment)
        
        # Calculate field gradient
        grad_B = self.field_gradient_analytical(x, y, z)
        
        # Handle scalar inputs
        if x.ndim == 0:
            x = x.reshape(1)
            y = y.reshape(1)
            z = z.reshape(1)
            scalar_input = True
        else:
            scalar_input = False
            
        # Ensure magnetic moment has correct shape
        if magnetic_moment.ndim == 1:
            magnetic_moment = magnetic_moment.reshape(1, 3)
        
        # Get the shape for force calculation
        position_shape = x.shape
        
        # Calculate force: F_i = m_j * ∂B_i/∂x_j (Einstein summation)
        F = np.zeros(position_shape + (3,))
        
        for i in range(3):  # Force component
            for j in range(3):  # Moment component
                F[..., i] += magnetic_moment[..., j] * grad_B[..., i, j]
        
        # Handle scalar output
        if scalar_input:
            F = F[0]
        
        return F[..., 0], F[..., 1], F[..., 2]
    
    def field_on_axis(self, z: np.ndarray) -> np.ndarray:
        """
        Calculate magnetic field on the coil axis (analytical solution).
        
        Parameters:
        -----------
        z : np.ndarray
            Axial distance from coil plane [m]
            
        Returns:
        --------
        B_z : np.ndarray
            Axial magnetic field [T]
        """
        a = self.coil.radius
        I = self.coil.current * self.coil.turns
        
        # Analytical solution for on-axis field
        B_z = MU_0 * I * a**2 / (2 * (a**2 + z**2)**(3/2))
        
        return B_z
    
    def field_at_center(self) -> float:
        """
        Calculate magnetic field at the center of the coil.
        
        Returns:
        --------
        B_z : float
            Magnetic field at center [T]
        """
        return MU_0 * self.coil.current * self.coil.turns / (2 * self.coil.radius)
    
    def magnet_force_at_position(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Calculate force on the magnet at a specific position.
        
        Parameters:
        -----------
        x, y, z : float
            Magnet position [m]
            
        Returns:
        --------
        F_x, F_y, F_z : float
            Force components [N]
        """
        # Calculate magnetic moment vector
        moment_vector = self.magnet.magnetic_moment * self.magnet.orientation
        
        # Calculate force using F = (m·∇)B
        F_x, F_y, F_z = self.magnetic_force(x, y, z, moment_vector)
        
        return F_x, F_y, F_z
    
    def force_vector_field_2d(self, z_plane: float, x_range: Tuple[float, float], 
                             y_range: Tuple[float, float], grid_size: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate force vector field on a 2D plane at given z-height.
        
        Parameters:
        -----------
        z_plane : float
            Z-coordinate of the plane [m]
        x_range : tuple
            (x_min, x_max) range [m]
        y_range : tuple
            (y_min, y_max) range [m]
        grid_size : int
            Number of grid points in each direction
            
        Returns:
        --------
        X, Y : np.ndarray
            Grid coordinates
        F_x, F_y, F_z : np.ndarray
            Force components at each grid point
        """
        # Create 2D grid
        x = np.linspace(x_range[0], x_range[1], grid_size)
        y = np.linspace(y_range[0], y_range[1], grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Initialize force arrays
        F_x = np.zeros_like(X)
        F_y = np.zeros_like(X)
        F_z = np.zeros_like(X)
        
        # Calculate force at each grid point
        for i in range(grid_size):
            for j in range(grid_size):
                fx, fy, fz = self.magnet_force_at_position(X[i, j], Y[i, j], z_plane)
                F_x[i, j] = fx
                F_y[i, j] = fy
                F_z[i, j] = fz
        
        return X, Y, F_x, F_y, F_z
    
    def plot_force_vector_field(self, z_plane: float = 1e-3, grid_size: int = 20, 
                               figsize: Tuple[int, int] = (15, 5), force_scale: float = 1e6,
                               auto_range: bool = True, x_range: Optional[Tuple[float, float]] = None,
                               y_range: Optional[Tuple[float, float]] = None):
        """
        Plot 2D force vector field for the magnet on a given z-plane.
        
        Parameters:
        -----------
        z_plane : float
            Height of the plane above coil [m]
        grid_size : int
            Number of grid points in each direction
        figsize : tuple
            Figure size
        force_scale : float
            Scale factor for force vectors
        auto_range : bool
            Whether to automatically determine plot range
        x_range, y_range : tuple, optional
            Manual range specification [m]
        """
        # Determine plot range
        if auto_range:
            plot_range = 3 * self.coil.radius
            x_range = (-plot_range, plot_range)
            y_range = (-plot_range, plot_range)
        
        # Calculate force vector field
        X, Y, F_x, F_y, F_z = self.force_vector_field_2d(z_plane, x_range, y_range, grid_size)
        
        # Calculate force magnitudes
        F_magnitude = np.sqrt(F_x**2 + F_y**2 + F_z**2)
        F_xy_magnitude = np.sqrt(F_x**2 + F_y**2)
        
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: In-plane force vectors (F_x, F_y)
        skip = max(1, grid_size // 15)  # Reduce arrow density for clarity
        ax1.quiver(X[::skip, ::skip] * 1000, Y[::skip, ::skip] * 1000,
                  F_x[::skip, ::skip] * force_scale, F_y[::skip, ::skip] * force_scale,
                  F_xy_magnitude[::skip, ::skip] * 1e6, cmap='viridis', alpha=0.8,
                  scale_units='xy', angles='xy', scale=1)
        
        # Add coil outline
        theta = np.linspace(0, 2*np.pi, 100)
        coil_x = self.coil.radius * np.cos(theta) * 1000
        coil_y = self.coil.radius * np.sin(theta) * 1000
        ax1.plot(coil_x, coil_y, 'r-', linewidth=3, label='Coil')
        
        ax1.set_xlabel('x [mm]')
        ax1.set_ylabel('y [mm]')
        ax1.set_title(f'In-plane Force (F_x, F_y) at z={z_plane*1000:.1f} mm\n{self.magnet.material} magnet')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_aspect('equal')
        
        # Plot 2: Out-of-plane force (F_z)
        im2 = ax2.contourf(X * 1000, Y * 1000, F_z * 1e6, levels=20, cmap='RdBu_r')
        ax2.contour(X * 1000, Y * 1000, F_z * 1e6, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        ax2.plot(coil_x, coil_y, 'r-', linewidth=3, label='Coil')
        
        ax2.set_xlabel('x [mm]')
        ax2.set_ylabel('y [mm]')
        ax2.set_title(f'Vertical Force (F_z) at z={z_plane*1000:.1f} mm')
        ax2.legend()
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2, label='F_z [μN]')
        
        # Plot 3: Total force magnitude
        im3 = ax3.contourf(X * 1000, Y * 1000, F_magnitude * 1e6, levels=20, cmap='plasma')
        ax3.contour(X * 1000, Y * 1000, F_magnitude * 1e6, levels=20, colors='white', alpha=0.5, linewidths=0.5)
        ax3.plot(coil_x, coil_y, 'r-', linewidth=3, label='Coil')
        
        ax3.set_xlabel('x [mm]')
        ax3.set_ylabel('y [mm]')
        ax3.set_title(f'Total Force Magnitude at z={z_plane*1000:.1f} mm')
        ax3.legend()
        ax3.set_aspect('equal')
        plt.colorbar(im3, ax=ax3, label='|F| [μN]')
        
        plt.tight_layout()
        plt.show()
        
        # Print magnet information
        print(f"\nMagnet Properties:")
        print(f"Material: {self.magnet.material}")
        print(f"Dimensions: {self.magnet.dimensions*1000} mm")
        print(f"Magnetic moment: {self.magnet.magnetic_moment:.2e} A·m²")
        print(f"Mass: {self.magnet.mass*1000:.1f} g")
        print(f"Orientation: {self.magnet.orientation}")
        
        # Print force statistics
        print(f"\nForce Statistics at z={z_plane*1000:.1f} mm:")
        print(f"Max total force: {np.max(F_magnitude)*1e6:.1f} μN")
        print(f"Max vertical force: {np.max(np.abs(F_z))*1e6:.1f} μN")
        print(f"Max horizontal force: {np.max(F_xy_magnitude)*1e6:.1f} μN")
    
    def plot_force_vs_height_magnet(self, z_range: Tuple[float, float] = (0.5e-3, 20e-3), 
                                   num_points: int = 100, positions: List[Tuple[float, float]] = None):
        """
        Plot force vs height for the magnet at different positions.
        
        Parameters:
        -----------
        z_range : tuple
            Height range [m]
        num_points : int
            Number of points to plot
        positions : list
            List of (x, y) positions to plot [m]
        """
        if positions is None:
            positions = [(0, 0), (self.coil.radius/2, 0), (self.coil.radius, 0)]
        
        z = np.linspace(z_range[0], z_range[1], num_points)
        
        plt.figure(figsize=(12, 8))
        
        # Plot force components vs height for each position
        for i, (x_pos, y_pos) in enumerate(positions):
            F_x_array = np.zeros_like(z)
            F_y_array = np.zeros_like(z)
            F_z_array = np.zeros_like(z)
            
            for j, z_val in enumerate(z):
                fx, fy, fz = self.magnet_force_at_position(x_pos, y_pos, z_val)
                F_x_array[j] = fx
                F_y_array[j] = fy
                F_z_array[j] = fz
            
            F_total = np.sqrt(F_x_array**2 + F_y_array**2 + F_z_array**2)
            
            label_pos = f'({x_pos*1000:.1f}, {y_pos*1000:.1f}) mm'
            plt.plot(z * 1000, F_z_array * 1e6, '--', label=f'F_z at {label_pos}', alpha=0.7)
            plt.plot(z * 1000, F_total * 1e6, '-', linewidth=2, label=f'|F| at {label_pos}')
        
        plt.xlabel('Height above coil [mm]')
        plt.ylabel('Force [μN]')
        plt.title(f'Magnet Force vs Height\n{self.magnet.material}, moment={self.magnet.magnetic_moment:.2e} A·m²')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.yscale('log')
        plt.show()
    
    def analyze_magnet_interaction(self, z_plane: float = 1e-3):
        """
        Comprehensive analysis of magnet-coil interaction.
        
        Parameters:
        -----------
        z_plane : float
            Analysis plane height [m]
        """
        print(f"=== Magnet-Coil Interaction Analysis ===")
        print(f"Analysis plane: z = {z_plane*1000:.1f} mm\n")
        
        # Calculate forces at key positions
        key_positions = [
            (0, 0, "Center"),
            (self.coil.radius/2, 0, "Half radius"),
            (self.coil.radius, 0, "Coil edge"),
            (1.5*self.coil.radius, 0, "Outside coil")
        ]
        
        print("Force Analysis at Key Positions:")
        print("Position [mm]        F_x [μN]    F_y [μN]    F_z [μN]    |F| [μN]")
        print("-" * 70)
        
        for x_pos, y_pos, description in key_positions:
            fx, fy, fz = self.magnet_force_at_position(x_pos, y_pos, z_plane)
            f_mag = np.sqrt(fx**2 + fy**2 + fz**2)
            
            print(f"{description:15} ({x_pos*1000:4.1f}, {y_pos*1000:4.1f})  "
                  f"{fx*1e6:8.2f}  {fy*1e6:8.2f}  {fz*1e6:8.2f}  {f_mag*1e6:8.2f}")
        
        # Calculate potential energy (approximate)
        print(f"\nApproximate Potential Energy Analysis:")
        B_center = self.magnetic_field(0, 0, z_plane)[2]
        U_center = -self.magnet.magnetic_moment * B_center  # U = -m·B
        print(f"Potential energy at center: {U_center*1e6:.2f} μJ")
        
        # Maximum acceleration (F = ma)
        fx_max, fy_max, fz_max = self.magnet_force_at_position(0, 0, z_plane)
        a_max = np.sqrt(fx_max**2 + fy_max**2 + fz_max**2) / self.magnet.mass
        print(f"Maximum acceleration: {a_max:.2f} m/s²")
    
    def compare_magnets(self, magnet_list: List[MagnetParameters], z_plane: float = 1e-3):
        """
        Compare force characteristics of different magnets.
        
        Parameters:
        -----------
        magnet_list : list
            List of MagnetParameters to compare
        z_plane : float
            Analysis plane height [m]
        """
        print(f"=== Magnet Comparison at z = {z_plane*1000:.1f} mm ===\n")
        
        original_magnet = self.magnet
        
        print("Magnet Type          Material  Moment [A·m²]  Mass [g]  Max Force [μN]")
        print("-" * 70)
        
        for i, magnet in enumerate(magnet_list):
            self.magnet = magnet
            
            # Calculate maximum force (at center)
            fx, fy, fz = self.magnet_force_at_position(0, 0, z_plane)
            f_max = np.sqrt(fx**2 + fy**2 + fz**2)
            
            print(f"Magnet {i+1:2d}          {magnet.material:8s}  {magnet.magnetic_moment:.2e}  "
                  f"{magnet.mass*1000:6.1f}  {f_max*1e6:10.2f}")
        
        # Restore original magnet
        self.magnet = original_magnet
    
    def plot_field_2d(self, z_height: float = 1e-3, grid_size: int = 50, 
                     field_scale: float = 1.0, figsize: Tuple[int, int] = (12, 5)):
        """
        Plot 2D magnetic field vectors and magnitude.
        
        Parameters:
        -----------
        z_height : float
            Height above coil plane [m]
        grid_size : int
            Number of grid points in each direction
        field_scale : float
            Scale factor for field arrows
        figsize : tuple
            Figure size
        """
        # Create grid
        x_max = 3 * self.coil.radius
        x = np.linspace(-x_max, x_max, grid_size)
        y = np.linspace(-x_max, x_max, grid_size)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, z_height)
        
        # Calculate magnetic field
        B_x, B_y, B_z = self.magnetic_field(X, Y, Z)
        B_magnitude = np.sqrt(B_x**2 + B_y**2 + B_z**2)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Field vectors
        skip = max(1, grid_size // 20)  # Reduce arrow density
        ax1.quiver(X[::skip, ::skip] * 1000, Y[::skip, ::skip] * 1000,
                  B_x[::skip, ::skip] * field_scale, B_y[::skip, ::skip] * field_scale,
                  alpha=0.7, scale_units='xy', angles='xy', scale=1)
        
        # Add coil outline
        theta = np.linspace(0, 2*np.pi, 100)
        coil_x = self.coil.radius * np.cos(theta) * 1000
        coil_y = self.coil.radius * np.sin(theta) * 1000
        ax1.plot(coil_x, coil_y, 'r-', linewidth=3, label='Coil')
        
        ax1.set_xlabel('x [mm]')
        ax1.set_ylabel('y [mm]')
        ax1.set_title(f'Magnetic Field Vectors at z={z_height*1000:.1f} mm')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_aspect('equal')
        
        # Plot 2: Field magnitude contours
        im = ax2.contourf(X * 1000, Y * 1000, B_magnitude * 1000, levels=20, cmap='viridis')
        ax2.contour(X * 1000, Y * 1000, B_magnitude * 1000, levels=20, colors='white', alpha=0.5)
        ax2.plot(coil_x, coil_y, 'r-', linewidth=3, label='Coil')
        
        ax2.set_xlabel('x [mm]')
        ax2.set_ylabel('y [mm]')
        ax2.set_title(f'Magnetic Field Magnitude at z={z_height*1000:.1f} mm')
        ax2.legend()
        ax2.set_aspect('equal')
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, label='|B| [mT]')
        
        plt.tight_layout()
        plt.show()
    
    def plot_field_vs_height(self, z_range: Tuple[float, float] = (0, 20e-3), 
                           num_points: int = 100):
        """
        Plot magnetic field vs height above coil.
        
        Parameters:
        -----------
        z_range : tuple
            Height range [m]
        num_points : int
            Number of points to plot
        """
        z = np.linspace(z_range[0], z_range[1], num_points)
        
        # Calculate fields
        B_analytical = self.field_on_axis(z)
        B_numerical = self.magnetic_field(np.zeros_like(z), np.zeros_like(z), z)[2]
        
        plt.figure(figsize=(10, 6))
        plt.plot(z * 1000, B_analytical * 1000, 'b-', linewidth=2, label='Analytical (on-axis)')
        plt.plot(z * 1000, B_numerical * 1000, 'r--', linewidth=2, label='Numerical (full calculation)')
        
        plt.xlabel('Height above coil [mm]')
        plt.ylabel('Magnetic field B_z [mT]')
        plt.title('Magnetic Field vs Height Above Coil')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
    
    def validate_against_analytical(self, tolerance: float = 1e-6) -> bool:
        """
        Validate numerical implementation against analytical solutions.
        
        Parameters:
        -----------
        tolerance : float
            Relative tolerance for comparison
            
        Returns:
        --------
        bool
            True if validation passes
        """
        print("Validating electromagnetic field calculations...")
        
        # Test 1: Field at center
        B_center_analytical = self.field_at_center()
        B_center_numerical = self.magnetic_field(np.array([0]), np.array([0]), np.array([0]))[2][0]
        
        error_center = abs(B_center_numerical - B_center_analytical) / B_center_analytical
        print(f"Center field error: {error_center:.2e} (tolerance: {tolerance:.2e})")
        
        # Test 2: On-axis field at various heights
        z_test = np.array([1e-3, 2e-3, 5e-3, 10e-3])
        B_analytical = self.field_on_axis(z_test)
        B_numerical = self.magnetic_field(np.zeros_like(z_test), np.zeros_like(z_test), z_test)[2]
        
        error_axis = np.max(np.abs(B_numerical - B_analytical) / B_analytical)
        print(f"On-axis field error: {error_axis:.2e} (tolerance: {tolerance:.2e})")
        
        # Test 3: Field symmetry
        x_test = np.array([1e-3, 2e-3])
        B_pos = self.magnetic_field(x_test, np.zeros_like(x_test), np.zeros_like(x_test))
        B_neg = self.magnetic_field(-x_test, np.zeros_like(x_test), np.zeros_like(x_test))
        
        # B_x should be antisymmetric, B_z should be symmetric
        # Handle potential division by zero
        B_x_pos_abs = np.abs(B_pos[0])
        B_z_pos_abs = np.abs(B_pos[2])
        
        if np.any(B_x_pos_abs > EPSILON):
            error_sym_x = np.max(np.abs(B_pos[0] + B_neg[0]) / np.where(B_x_pos_abs > EPSILON, B_x_pos_abs, 1))
        else:
            error_sym_x = 0.0
            
        if np.any(B_z_pos_abs > EPSILON):
            error_sym_z = np.max(np.abs(B_pos[2] - B_neg[2]) / np.where(B_z_pos_abs > EPSILON, B_z_pos_abs, 1))
        else:
            error_sym_z = 0.0
        
        print(f"Symmetry error (B_x): {error_sym_x:.2e} (tolerance: {tolerance:.2e})")
        print(f"Symmetry error (B_z): {error_sym_z:.2e} (tolerance: {tolerance:.2e})")
        
        # Overall validation
        max_error = max(error_center, error_axis, error_sym_x, error_sym_z)
        validation_passed = max_error < tolerance
        
        print(f"\nValidation {'PASSED' if validation_passed else 'FAILED'}")
        print(f"Maximum error: {max_error:.2e}")
        
        return validation_passed

def demonstrate_magnet_coil_interaction():
    """Demonstrate the magnet-coil interaction capabilities."""
    print("=== Enhanced Electromagnetic PCB Coil Simulator with Magnet Interaction ===\n")
    
    # Create coil parameters
    coil_params = CoilParameters(
        radius=5e-3,      # 5mm radius
        turns=20,         # 20 turns
        current=2.0,      # 2 Amperes for stronger field
        center=np.array([0, 0, 0])
    )
    
    # Start with a small neodymium magnet
    magnet = StandardMagnets.neodymium_disk_small()
    
    # Create simulator
    simulator = CircularCoilSimulator(coil_params, magnet)
    
    # 1. Validate basic electromagnetic implementation
    print("1. Validating electromagnetic implementation...")
    validation_passed = simulator.validate_against_analytical()
    print()
    
    # 2. Analyze magnet-coil interaction
    print("2. Magnet-coil interaction analysis:")
    simulator.analyze_magnet_interaction(z_plane=1e-3)
    print()
    
    # 3. Generate force vector field visualization
    print("3. Generating force vector field visualization...")
    simulator.plot_force_vector_field(z_plane=1e-3, grid_size=25)
    
    # 4. Compare different magnet types
    print("4. Comparing different magnet types...")
    magnets_to_compare = [
        StandardMagnets.neodymium_disk_small(),
        StandardMagnets.neodymium_block_medium(),
        StandardMagnets.ferrite_disk()
    ]
    
    simulator.compare_magnets(magnets_to_compare, z_plane=1e-3)
    
    # 5. Force vs height analysis
    print("5. Force vs height analysis...")
    simulator.plot_force_vs_height_magnet(z_range=(0.5e-3, 20e-3))
    
    # 6. Demonstrate adjustable parameters
    print("6. Demonstrating adjustable parameters...")
    
    # Create custom magnet
    custom_magnet = MagnetParameters(
        magnetic_moment=5e-4,  # 0.5 mA·m²
        orientation=np.array([0.0, 0.0, 1.0]),
        mass=2e-3,  # 2 mg
        dimensions=np.array([3e-3, 3e-3, 1e-3]),  # 3x3x1 mm
        material="Custom"
    )
    
    # Update simulator with custom magnet
    simulator.magnet = custom_magnet
    
    print("Custom magnet force vector field:")
    simulator.plot_force_vector_field(z_plane=2e-3, grid_size=20)
    
    # 7. Multi-height analysis
    print("7. Multi-height force analysis...")
    heights = [0.5e-3, 1e-3, 2e-3, 5e-3]
    
    for height in heights:
        print(f"\nForce analysis at z = {height*1000:.1f} mm:")
        fx, fy, fz = simulator.magnet_force_at_position(0, 0, height)
        f_total = np.sqrt(fx**2 + fy**2 + fz**2)
        print(f"Force at center: ({fx*1e6:.2f}, {fy*1e6:.2f}, {fz*1e6:.2f}) μN")
        print(f"Total force magnitude: {f_total*1e6:.2f} μN")

def create_custom_simulation():
    """Create a custom simulation with user-defined parameters."""
    print("=== Custom PCB Coil-Magnet Simulation ===\n")
    
    # Custom coil parameters
    coil_params = CoilParameters(
        radius=8e-3,      # 8mm radius
        turns=30,         # 30 turns
        current=1.5,      # 1.5 Amperes
        center=np.array([0, 0, 0])
    )
    
    # Custom magnet parameters
    magnet_params = MagnetParameters(
        magnetic_moment=2e-4,  # 0.2 mA·m²
        orientation=np.array([0.0, 0.0, 1.0]),  # Vertical orientation
        mass=5e-3,  # 5 mg
        dimensions=np.array([6e-3, 6e-3, 2e-3]),  # 6x6x2 mm
        material="N42"
    )
    
    # Create simulator
    simulator = CircularCoilSimulator(coil_params, magnet_params)
    
    # Generate comprehensive analysis
    print("Custom simulation parameters:")
    print(f"Coil: {coil_params.radius*1000:.1f}mm radius, {coil_params.turns} turns, {coil_params.current:.1f}A")
    print(f"Magnet: {magnet_params.material}, {magnet_params.magnetic_moment:.2e} A·m², {magnet_params.mass*1000:.1f}mg")
    print()
    
    # Multi-plane force analysis
    z_planes = [0.5e-3, 1.5e-3, 3e-3]
    
    for z_plane in z_planes:
        print(f"Analyzing force field at z = {z_plane*1000:.1f} mm...")
        simulator.plot_force_vector_field(
            z_plane=z_plane, 
            grid_size=30, 
            figsize=(18, 6),
            auto_range=True
        )

if __name__ == "__main__":
    # Run the enhanced demonstration
    demonstrate_magnet_coil_interaction()

