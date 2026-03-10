"""
Synthetic Dataset Module for Loss Landscape Demo.

This module provides synthetic datasets suitable for demonstrating
loss landscape visualization in deep learning optimization.
"""

import numpy as np
import torch
from typing import Tuple, Optional


class SineCurveFittingDataset(torch.utils.data.Dataset):
    """
    Synthetic dataset for sine curve fitting.

    The task is to fit a noisy sine curve using neural networks.
    This is a classic problem that exhibits interesting loss landscape dynamics.

    y = sin(x) + noise
    """

    def __init__(
        self,
        num_samples: int = 100,
        x_range: Tuple[float, float] = (-4 * np.pi, 4 * np.pi),
        noise_std: float = 0.2,
        input_dim: int = 1,
    ):
        """
        Initialize the sine curve fitting dataset.

        Args:
            num_samples: Number of data points
            x_range: (min, max) range for x values
            noise_std: Standard deviation of Gaussian noise
            input_dim: Dimension of input features (for compatibility)
        """
        self.num_samples = num_samples
        self.x_range = x_range
        self.noise_std = noise_std
        self.input_dim = input_dim

        # Generate data
        self.X, self.y = self._generate_data()

    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sine curve with noise."""
        # Generate x values uniformly
        x = torch.linspace(
            self.x_range[0], self.x_range[1], self.num_samples
        )

        # Add random permutation to create pairs
        indices = torch.randperm(self.num_samples)
        x_sampled = x[indices]

        # Compute y = sin(x) + noise
        noise = self.noise_std * torch.randn_like(x_sampled)
        y = torch.sin(x_sampled) + noise

        # Reshape for compatibility
        X = x_sampled.unsqueeze(-1)  # (N, 1)
        return X, y

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_sine_dataset(
    num_train: int = 100,
    num_test: int = 50,
    x_range: Tuple[float, float] = (-4 * np.pi, 4 * np.pi),
    noise_std: float = 0.2,
    batch_size: int = 32,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training and test data loaders for sine curve fitting.

    Args:
        num_train: Number of training samples
        num_test: Number of test samples
        x_range: (min, max) range for x values
        noise_std: Noise standard deviation
        batch_size: Batch size for data loaders

    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset = SineCurveFittingDataset(
        num_samples=num_train,
        x_range=x_range,
        noise_std=noise_std,
    )
    test_dataset = SineCurveFittingDataset(
        num_samples=num_test,
        x_range=x_range,
        noise_std=noise_std,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader


class PolynomialFittingDataset(torch.utils.data.Dataset):
    """
    Synthetic dataset for polynomial fitting.

    The task is to fit a noisy polynomial curve.
    This demonstrates multi-modal loss landscapes.
    """

    def __init__(
        self,
        num_samples: int = 100,
        x_range: Tuple[float, float] = (-3, 3),
        noise_std: float = 0.3,
        degree: int = 3,
    ):
        """
        Initialize the polynomial fitting dataset.

        Args:
            num_samples: Number of data points
            x_range: (min, max) range for x values
            noise_std: Standard deviation of Gaussian noise
            degree: Degree of the polynomial
        """
        self.num_samples = num_samples
        self.x_range = x_range
        self.noise_std = noise_std
        self.degree = degree

        # Generate data
        self.X, self.y = self._generate_data()

    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate polynomial curve with noise."""
        x = torch.linspace(self.x_range[0], self.x_range[1], self.num_samples)
        indices = torch.randperm(self.num_samples)
        x_sampled = x[indices]

        # Generate random coefficients for polynomial
        coeffs = torch.randn(self.degree + 1)

        # Compute y = sum(coeffs[i] * x^i) + noise
        y = torch.zeros_like(x_sampled)
        for i in range(self.degree + 1):
            y += coeffs[i] * (x_sampled ** i)

        # Add noise
        y = y + self.noise_std * torch.randn_like(y)

        X = x_sampled.unsqueeze(-1)
        return X, y

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_polynomial_dataset(
    num_train: int = 100,
    num_test: int = 50,
    x_range: Tuple[float, float] = (-3, 3),
    noise_std: float = 0.3,
    degree: int = 3,
    batch_size: int = 32,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training and test data loaders for polynomial fitting.

    Args:
        num_train: Number of training samples
        num_test: Number of test samples
        x_range: (min, max) range for x values
        noise_std: Noise standard deviation
        degree: Polynomial degree
        batch_size: Batch size

    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset = PolynomialFittingDataset(
        num_samples=num_train,
        x_range=x_range,
        noise_std=noise_std,
        degree=degree,
    )
    test_dataset = PolynomialFittingDataset(
        num_samples=num_test,
        x_range=x_range,
        noise_std=noise_std,
        degree=degree,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader


# Pre-computed landscapes directory
PRECOMPUTED_LANDSCAPES_DIR = "data/precomputed_landscapes"


def get_precomputed_landscapes_path() -> str:
    """Get the path to precomputed landscapes directory."""
    import os
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return os.path.join(project_root, PRECOMPUTED_LANDSCAPES_DIR)


if __name__ == "__main__":
    # Test the datasets
    print("Testing SineCurveFittingDataset...")
    train_loader, test_loader = create_sine_dataset(
        num_train=100, num_test=50, batch_size=32
    )
    print(f"Train loader: {len(train_loader.dataset)} samples")
    print(f"Test loader: {len(test_loader.dataset)} samples")

    # Check first batch
    for X, y in train_loader:
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        break
