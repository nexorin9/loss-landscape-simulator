"""
Loss Landscape Computation Module.

This module provides functionality to compute loss values over a 2D grid
around current parameters, enabling visualization of neural network optimization dynamics.
"""

import os
import pickle
from typing import Callable, Optional, Tuple

import numpy as np
import torch


class LossLandscape:
    """
    Computes and manages loss landscape data for neural network visualization.

    The loss landscape is computed by interpolating model parameters in 2D space
    around the current parameter values and evaluating the loss at each point.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        data_loader: Optional[torch.utils.data.DataLoader] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the LossLandscape calculator.

        Args:
            model: The neural network model
            criterion: Loss function that takes (predictions, targets) -> loss
            data_loader: Data loader for getting batches. If None, uses dummy data.
            cache_dir: Directory to save/load cached landscapes. If None, caching is disabled.
        """
        self.model = model
        self.criterion = criterion
        self.data_loader = data_loader
        self.cache_dir = cache_dir

        # Cache for computed landscapes
        self._landscape_cache: dict = {}

    def interpolate_params(
        self,
        base_params: torch.Tensor,
        direction1: torch.Tensor,
        direction2: torch.Tensor,
        grid_size: int = 20,
        param_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> torch.Tensor:
        """
        Interpolate model parameters over a 2D grid.

        Args:
            base_params: Base parameter vector (1D tensor)
            direction1: First interpolation direction (1D tensor, same shape as base_params)
            direction2: Second interpolation direction (1D tensor, same shape as base_params)
            grid_size: Number of points along each axis
            param_range: (min, max) range for interpolation

        Returns:
            Tensor of shape (grid_size, grid_size, num_params) containing interpolated parameters
        """
        # Generate interpolation coefficients
        coeffs = np.linspace(param_range[0], param_range[1], grid_size)
        coeff_grid = np.meshgrid(coeffs, coeffs)

        # Create parameter grid
        param_grid = torch.zeros(grid_size, grid_size, base_params.numel())

        for i in range(grid_size):
            for j in range(grid_size):
                alpha = coeff_grid[0][i, j]
                beta = coeff_grid[1][i, j]
                param_grid[i, j] = base_params + alpha * direction1 + beta * direction2

        return param_grid

    def compute_loss_surface(
        self,
        grid_size: int = 20,
        param_range: Tuple[float, float] = (-1.0, 1.0),
        direction1: Optional[torch.Tensor] = None,
        direction2: Optional[torch.Tensor] = None,
        cache_key: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the loss surface over a 2D grid around current parameters.

        Args:
            grid_size: Number of points along each axis
            param_range: (min, max) range for interpolation
            direction1: First interpolation direction. If None, uses random orthogonal directions.
            direction2: Second interpolation direction. If None, uses random orthogonal directions.
            cache_key: Key for caching. If provided, saves/loads from cache.

        Returns:
            Tuple of (loss_surface, alpha_grid, beta_grid) where:
                - loss_surface: 2D array of shape (grid_size, grid_size)
                - alpha_grid: 2D array of alpha coefficients
                - beta_grid: 2D array of beta coefficients
        """
        # Check cache first
        if cache_key is not None and self._try_load_cache(cache_key) is not None:
            cached = self._try_load_cache(cache_key)
            if cached is not None:
                return cached

        # Get current parameters
        current_params = self.model.get_flat_params().detach()

        # Generate interpolation directions if not provided
        if direction1 is None:
            direction1 = torch.randn_like(current_params)
            direction1 = direction1 / torch.norm(direction1)

        if direction2 is None:
            # Generate orthogonal direction
            direction2 = torch.randn_like(current_params)
            direction2 = direction2 - torch.dot(direction2, direction1) * direction1
            direction2 = direction2 / torch.norm(direction2)

        # Interpolate parameters over the grid
        param_grid = self.interpolate_params(
            current_params, direction1, direction2, grid_size, param_range
        )

        # Compute loss at each point
        loss_surface = np.zeros((grid_size, grid_size))

        with torch.no_grad():
            for i in range(grid_size):
                for j in range(grid_size):
                    # Set parameters
                    self.model.set_flat_params(param_grid[i, j])

                    # Get a batch of data
                    if self.data_loader is not None:
                        try:
                            inputs, targets = next(iter(self.data_loader))
                        except StopIteration:
                            # Reset loader if exhausted
                            self.data_loader = torch.utils.data.DataLoader(
                                self.data_loader.dataset,
                                batch_size=self.data_loader.batch_size,
                                shuffle=True,
                            )
                            inputs, targets = next(iter(self.data_loader))
                    else:
                        # Use dummy data
                        inputs = torch.randn(4, 2)
                        targets = torch.randn(4, 1)

                    # Compute loss
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss_surface[i, j] = loss.item()

        # Generate coordinate grids for return values
        alpha_vals = np.linspace(param_range[0], param_range[1], grid_size)
        beta_vals = np.linspace(param_range[0], param_range[1], grid_size)
        alpha_grid, beta_grid = np.meshgrid(alpha_vals, beta_vals)

        # Store in cache if key provided
        if cache_key is not None and self.cache_dir is not None:
            self._save_cache(cache_key, (loss_surface, param_range, param_range))

        return loss_surface, alpha_grid, beta_grid

    def _get_cache_path(self, key: str) -> str:
        """Get the full path for a cache file."""
        if self.cache_dir is None:
            raise ValueError("Cache directory not set")
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"{key}.pkl")

    def _save_cache(
        self,
        key: str,
        data: Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]],
    ) -> None:
        """Save landscape data to cache."""
        try:
            cache_path = self._get_cache_path(key)
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
        except Exception:
            pass  # Silently fail if caching fails

    def _try_load_cache(
        self,
        key: str,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Try to load cached landscape data."""
        if self.cache_dir is None:
            return None

        try:
            cache_path = self._get_cache_path(key)
            with open(cache_path, "rb") as f:
                data = pickle.load(f)

            loss_surface, alpha_range, beta_range = data
            alpha_vals = np.linspace(alpha_range[0], alpha_range[1], len(loss_surface))
            beta_vals = np.linspace(beta_range[0], beta_range[1], len(loss_surface))
            alpha_grid, beta_grid = np.meshgrid(alpha_vals, beta_vals)

            return loss_surface, alpha_grid, beta_grid
        except Exception:
            return None


def compute_random_directions(
    model: torch.nn.Module,
    num_directions: int = 2,
) -> list[torch.Tensor]:
    """
    Compute random orthogonal directions for loss landscape interpolation.

    Args:
        model: The neural network model
        num_directions: Number of directions to generate

    Returns:
        List of orthonormal direction tensors
    """
    current_params = model.get_flat_params().detach()
    directions = []

    for i in range(num_directions):
        # Generate random direction
        direction = torch.randn_like(current_params)

        # Orthonormalize against existing directions (Gram-Schmidt)
        for existing_dir in directions:
            direction = direction - torch.dot(direction, existing_dir) * existing_dir

        # Normalize
        norm = torch.norm(direction)
        if norm > 1e-8:
            direction = direction / norm
            directions.append(direction)

    return directions


def create_dummy_data_loader(
    batch_size: int = 4,
    input_dim: int = 2,
    output_dim: int = 1,
) -> torch.utils.data.DataLoader:
    """
    Create a dummy data loader for loss computation when no real data is available.

    Args:
        batch_size: Batch size
        input_dim: Input dimension
        output_dim: Output dimension

    Returns:
        DataLoader yielding random (inputs, targets) batches
    """

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples: int = 100):
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return torch.randn(input_dim), torch.randn(output_dim)

    dataset = DummyDataset(num_samples=100)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    # Test the LossLandscape class
    from model import SimpleMLP

    # Create a simple model and loss function
    model = SimpleMLP(input_dim=2, hidden_dim=10, output_dim=1)
    criterion = torch.nn.MSELoss()

    # Create dummy data loader
    data_loader = create_dummy_data_loader(batch_size=4, input_dim=2, output_dim=1)

    # Initialize loss landscape calculator
    landscape = LossLandscape(
        model=model,
        criterion=criterion,
        data_loader=data_loader,
        cache_dir=None,  # Disable caching for test
    )

    # Compute loss surface
    loss_surface, alpha_grid, beta_grid = landscape.compute_loss_surface(
        grid_size=10, param_range=(-0.5, 0.5)
    )

    print(f"Loss surface shape: {loss_surface.shape}")
    print(f"Alpha grid shape: {alpha_grid.shape}")
    print(f"Beta grid shape: {beta_grid.shape}")
    print(f"Loss range: [{loss_surface.min():.4f}, {loss_surface.max():.4f}]")
