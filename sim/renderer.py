"""
Matplotlib 3D Visualization Module for Loss Landscape.

This module provides visualization functions to display the loss landscape
as a 3D surface, with optional contour lines and trajectory overlay.
"""

import os

# Use non-interactive backend for headless environments
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import Optional


class Renderer:
    """
    Renders loss landscape visualizations using matplotlib 3D.

    Supports 3D surface plots with optional contour lines, and
    provides methods for labeling, saving, and displaying the plot.
    """

    def __init__(
        self,
        figsize: tuple = (10, 8),
        elevation: float = 30.0,
        azimuth: float = -45.0,
    ):
        """
        Initialize the renderer.

        Args:
            figsize: Figure size as (width, height)
            elevation: Elevation angle for 3D view
            azimuth: Azimuth angle for 3D view
        """
        self.figsize = figsize
        self.elevation = elevation
        self.azimuth = azimuth

        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[Axes3D] = None
        self._is_initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure the figure and axes are initialized."""
        if not self._is_initialized:
            self.init_figure()

    def init_figure(self) -> None:
        """Initialize the matplotlib figure and 3D axes."""
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._is_initialized = True

    def plot_3d_surface(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        alpha: float = 0.7,
        cmap: str = "viridis",
        contour_lines: bool = False,
        n_contours: int = 15,
    ) -> None:
        """
        Plot a 3D surface representing the loss landscape.

        Args:
            X: X coordinates grid (2D array)
            Y: Y coordinates grid (2D array)
            Z: Z values (loss) grid (2D array)
            alpha: Surface transparency
            cmap: Colormap for the surface
            contour_lines: Whether to add contour lines overlay
            n_contours: Number of contour levels to show
        """
        self._ensure_initialized()

        # Plot the surface
        surf = self.ax.plot_surface(
            X, Y, Z, alpha=alpha, cmap=cmap, edgecolor="none", antialiased=True
        )

        # Add contour lines if requested
        if contour_lines:
            self.ax.contour(
                X, Y, Z, zdir="z", offset=Z.min(), levels=n_contours, colors="k", linewidths=0.5
            )
            self.ax.contour(
                X, Y, Z, zdir="x", offset=X.min(), levels=n_contours, colors="k", linewidths=0.5
            )
            self.ax.contour(
                X, Y, Z, zdir="y", offset=Y.max(), levels=n_contours, colors="k", linewidths=0.5
            )

        # Add colorbar
        self.fig.colorbar(surf, ax=self.ax, shrink=0.6, aspect=20)

    def set_title(self, title: str) -> None:
        """Set the plot title."""
        self._ensure_initialized()
        self.ax.set_title(title, fontsize=14, fontweight="bold")

    def set_xlabel(self, label: str) -> None:
        """Set the x-axis label."""
        self._ensure_initialized()
        self.ax.set_xlabel(label)

    def set_ylabel(self, label: str) -> None:
        """Set the y-axis label."""
        self._ensure_initialized()
        self.ax.set_ylabel(label)

    def set_zlabel(self, label: str) -> None:
        """Set the z-axis label."""
        self._ensure_initialized()
        self.ax.set_zlabel(label)

    def add_trajectory(
        self,
        positions: np.ndarray,
        color: str = "red",
        marker_size: float = 50,
        line_width: float = 2.0,
        cmap_trajectory: bool = False,
        z_offset: Optional[float] = None,
    ) -> None:
        """
        Add a trajectory overlay to the plot.

        Args:
            positions: Array of shape (n_points, 2) with x,y coordinates
            color: Line color for the trajectory
            marker_size: Size of the particle marker
            line_width: Width of the trajectory line
            cmap_trajectory: Whether to use color gradient along trajectory
            z_offset: Optional z-offset for 3D plots (uses mid-z if None)
        """
        self._ensure_initialized()

        positions = np.atleast_2d(positions)
        n_points = positions.shape[0]

        if n_points == 0:
            return

        xs = positions[:, 0]
        ys = positions[:, 1]

        # Handle both 3D and 2D plots
        if self.ax.name == "3d":
            # For 3D plots, use z range or default to mid-z
            if z_offset is None:
                z_min, z_max = self.ax.get_zlim()
                z_offset = (z_min + z_max) / 2

            zs = np.full_like(xs, z_offset)

            if cmap_trajectory and n_points > 1:
                # Color gradient along trajectory in 3D
                for i in range(n_points - 1):
                    self.ax.plot(
                        [xs[i], xs[i + 1]],
                        [ys[i], ys[i + 1]],
                        [zs[i], zs[i + 1]],
                        color=color,
                        linewidth=line_width * (i / n_points + 0.2),
                        alpha=0.8,
                    )
            else:
                self.ax.plot(xs, ys, zs, color=color, linewidth=line_width, alpha=0.8)

            # Add marker at final position
            if n_points > 0:
                self.ax.scatter(
                    xs[-1], ys[-1], zs[-1], c=color, s=marker_size, marker="o", edgecolors="white"
                )
        else:
            # For 2D contour plots
            if cmap_trajectory and n_points > 1:
                # Color gradient along trajectory in 2D
                for i in range(n_points - 1):
                    self.ax.plot(
                        [xs[i], xs[i + 1]],
                        [ys[i], ys[i + 1]],
                        color=color,
                        linewidth=line_width * (i / n_points + 0.2),
                        alpha=0.8,
                    )
            else:
                self.ax.plot(xs, ys, color=color, linewidth=line_width, alpha=0.8)

            # Add marker at final position
            if n_points > 0:
                self.ax.scatter(xs[-1], ys[-1], c=color, s=marker_size, marker="o", edgecolors="white")

    def savefig(self, filepath: str) -> None:
        """
        Save the figure to a file.

        Args:
            filepath: Path to save the image
        """
        self._ensure_initialized()
        self.fig.savefig(filepath, dpi=150, bbox_inches="tight")

    def show(self) -> None:
        """Display the plot."""
        self._ensure_initialized()
        # Set default viewing angle if not set
        if self.elevation is not None and self.azimuth is not None:
            self.ax.view_init(elev=self.elevation, azim=self.azimuth)
        plt.tight_layout()
        # Use interactive show only when in interactive environment
        import sys

        if sys.stdin.isatty():
            plt.show()

    def show_or_save(self, filepath: Optional[str] = None) -> None:
        """
        Show the plot or save to file.

        Args:
            filepath: If provided, save to file instead of showing.
                     If None and in interactive environment, show the plot.
        """
        self._ensure_initialized()
        # Set default viewing angle if not set
        if self.elevation is not None and self.azimuth is not None:
            self.ax.view_init(elev=self.elevation, azim=self.azimuth)
        plt.tight_layout()

        if filepath is not None:
            self.savefig(filepath)
        else:
            import sys

            if sys.stdin.isatty():
                plt.show()

    def clear(self) -> None:
        """Clear the current axes for a new plot."""
        if self.ax is not None:
            self.ax.clear()
        # Re-initialize 3D projection if needed
        if self.fig is not None and self.ax is None:
            self.init_figure()

    def close(self) -> None:
        """Close the figure completely."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self._is_initialized = False

    def plot_contour(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        levels: int = 15,
        cmap: str = "viridis",
        add_colorbar: bool = True,
        alpha: float = 0.7,
    ) -> None:
        """
        Plot a 2D contour view of the loss landscape.

        Args:
            X: X coordinates grid (2D array)
            Y: Y coordinates grid (2D array)
            Z: Z values (loss) grid (2D array)
            levels: Number of contour levels
            cmap: Colormap for the contour plot
            add_colorbar: Whether to add a colorbar
            alpha: Transparency of the filled contours
        """
        self._ensure_initialized()

        # Create filled contour plot
        contour = self.ax.contourf(
            X, Y, Z,
            levels=levels,
            cmap=cmap,
            alpha=alpha,
        )

        # Add contour lines
        contour_lines = self.ax.contour(
            X, Y, Z,
            levels=levels,
            colors="black",
            linewidths=0.5,
        )

        # Add labels to contour lines
        self.ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")

        # Add colorbar if requested
        if add_colorbar:
            self.fig.colorbar(contour, ax=self.ax, shrink=0.6, aspect=20)

    def set_aspect_equal(self) -> None:
        """Set equal aspect ratio for 2D contour plots."""
        if not self._is_initialized:
            return
        self.ax.set_aspect("equal")

    def plot_vector_field(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
        scale: float = 1.0,
        color: str = "red",
        alpha: float = 0.6,
        length: float = 0.3,
    ) -> None:
        """
        Plot a vector field on the loss landscape.

        Args:
            X: X coordinates grid (2D array)
            Y: Y coordinates grid (2D array)
            U: Vector field x-components (2D array, same shape as X)
            V: Vector field y-components (2D array, same shape as Y)
            scale: Scale factor for arrow lengths
            color: Arrow color
            alpha: Arrow transparency
            length: Maximum arrow length
        """
        self._ensure_initialized()

        # Normalize vectors and create arrows
        # Compute magnitude for scaling
        magnitudes = np.sqrt(U**2 + V**2)

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            # Normalize vectors
            normalized_U = U / magnitudes
            normalized_V = V / magnitudes

            # Create arrows at each grid point
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    x_pos = X[i, j]
                    y_pos = Y[i, j]
                    u_mag = magnitudes[i, j]

                    if u_mag > 0:  # Only draw non-zero vectors
                        # Scale arrow length by magnitude
                        arrow_len = min(length * (u_mag / magnitudes.max()), length)

                        # Draw the arrow
                        self.ax.arrow(
                            x_pos,
                            y_pos,
                            normalized_U[i, j] * arrow_len,
                            normalized_V[i, j] * arrow_len,
                            color=color,
                            alpha=alpha,
                            head_width=arrow_len * 0.3,
                            head_length=arrow_len * 0.25,
                            length_includes_head=True,
                            overhang=0.2,
                        )


def create_3d_renderer(
    figsize: tuple = (10, 8),
    elevation: float = 30.0,
    azimuth: float = -45.0,
) -> Renderer:
    """
    Factory function to create a configured renderer.

    Args:
        figsize: Figure size as (width, height)
        elevation: Elevation angle for 3D view
        azimuth: Azimuth angle for 3D view

    Returns:
        Configured Renderer instance with axes initialized
    """
    renderer = Renderer(figsize=figsize, elevation=elevation, azimuth=azimuth)
    renderer.init_figure()
    return renderer


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    x = np.linspace(-2, 2, 30)
    y = np.linspace(-2, 2, 30)
    X, Y = np.meshgrid(x, y)

    # Simple quadratic loss landscape
    Z = X**2 + Y**2

    # Create renderer and plot
    renderer = create_3d_renderer(figsize=(10, 8))
    renderer.plot_3d_surface(X, Y, Z, alpha=0.8, cmap="viridis", contour_lines=True)
    renderer.set_title("Loss Landscape - Quadratic")
    renderer.set_xlabel("Parameter 1")
    renderer.set_ylabel("Parameter 2")
    renderer.set_zlabel("Loss")

    # Add a sample trajectory
    t = np.linspace(0, 1, 10)
    traj_x = 2 * (1 - t)
    traj_y = 2 * (1 - t)
    traj = np.column_stack([traj_x, traj_y])
    renderer.add_trajectory(traj, color="red", line_width=2.5)

    renderer.show()
