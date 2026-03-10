"""
Generate Pre-computed Loss Landscapes.

This script pre-computes loss landscapes for common models and saves them
as numpy files for quick loading during visualization.
"""

import os
import sys
import argparse
import torch
import numpy as np

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.model import SimpleMLP, SmallCNN
from src.landscape import LossLandscape, create_dummy_data_loader
from data.datasets.synthetic import create_sine_dataset


def compute_and_save_landscape(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    cache_dir: str,
    filename: str,
    grid_size: int = 30,
    param_range: tuple = (-1.0, 1.0),
) -> None:
    """
    Compute loss landscape and save to file.

    Args:
        model: The neural network model
        criterion: Loss function
        data_loader: Data loader for training data
        cache_dir: Directory to save the cached landscape
        filename: Filename for saved landscape
        grid_size: Number of points along each axis
        param_range: (min, max) range for interpolation
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Initialize landscape calculator
    landscape = LossLandscape(
        model=model,
        criterion=criterion,
        data_loader=data_loader,
        cache_dir=cache_dir,
    )

    # Compute loss surface
    print(f"Computing loss landscape for {filename}...")
    loss_surface, alpha_grid, beta_grid = landscape.compute_loss_surface(
        grid_size=grid_size,
        param_range=param_range,
    )

    # Save to file
    output_path = os.path.join(cache_dir, filename)
    np.savez_compressed(
        output_path,
        loss_surface=loss_surface,
        alpha_grid=alpha_grid,
        beta_grid=beta_grid,
    )
    print(f"Saved landscape to {output_path}")
    print(f"  Loss range: [{loss_surface.min():.4f}, {loss_surface.max():.4f}]")


def generate_sine_fitting_landscapes(cache_dir: str) -> None:
    """Generate landscapes for sine curve fitting with different models."""

    # Create data loader
    train_loader, _ = create_sine_dataset(
        num_train=200, num_test=50, batch_size=32
    )

    # Model 1: Small MLP (input_dim=1, hidden_dim=5, output_dim=1)
    model1 = SimpleMLP(input_dim=1, hidden_dim=5, output_dim=1)
    criterion = torch.nn.MSELoss()

    compute_and_save_landscape(
        model=model1,
        criterion=criterion,
        data_loader=train_loader,
        cache_dir=cache_dir,
        filename="sine_mlp_small.npz",
        grid_size=25,
        param_range=(-0.5, 0.5),
    )

    # Model 2: Medium MLP (input_dim=1, hidden_dim=10, output_dim=1)
    model2 = SimpleMLP(input_dim=1, hidden_dim=10, output_dim=1)

    compute_and_save_landscape(
        model=model2,
        criterion=criterion,
        data_loader=train_loader,
        cache_dir=cache_dir,
        filename="sine_mlp_medium.npz",
        grid_size=25,
        param_range=(-0.5, 0.5),
    )

    # Model 3: Large MLP (input_dim=1, hidden_dim=20, output_dim=1)
    model3 = SimpleMLP(input_dim=1, hidden_dim=20, output_dim=1)

    compute_and_save_landscape(
        model=model3,
        criterion=criterion,
        data_loader=train_loader,
        cache_dir=cache_dir,
        filename="sine_mlp_large.npz",
        grid_size=25,
        param_range=(-0.5, 0.5),
    )


def generate_polynomial_fitting_landscapes(cache_dir: str) -> None:
    """Generate landscapes for polynomial fitting."""

    train_loader = create_dummy_data_loader(
        batch_size=16, input_dim=2, output_dim=1
    )

    # Model: MLP for 2D input
    model = SimpleMLP(input_dim=2, hidden_dim=10, output_dim=1)
    criterion = torch.nn.MSELoss()

    compute_and_save_landscape(
        model=model,
        criterion=criterion,
        data_loader=train_loader,
        cache_dir=cache_dir,
        filename="polynomial_mlp.npz",
        grid_size=25,
        param_range=(-0.3, 0.3),
    )


def generate_random_loss_landscapes(cache_dir: str) -> None:
    """
    Generate landscapes for randomly initialized models.

    These demonstrate the typical loss landscape structure.
    """

    # Create dummy data loader
    train_loader = create_dummy_data_loader(
        batch_size=16, input_dim=2, output_dim=1
    )

    # Model 1: Simple MLP with small hidden size
    model1 = SimpleMLP(input_dim=2, hidden_dim=5, output_dim=1)
    criterion = torch.nn.MSELoss()

    compute_and_save_landscape(
        model=model1,
        criterion=criterion,
        data_loader=train_loader,
        cache_dir=cache_dir,
        filename="random_mlp_small.npz",
        grid_size=20,
        param_range=(-0.5, 0.5),
    )

    # Model 2: MLP with larger hidden size
    model2 = SimpleMLP(input_dim=2, hidden_dim=15, output_dim=1)

    compute_and_save_landscape(
        model=model2,
        criterion=criterion,
        data_loader=train_loader,
        cache_dir=cache_dir,
        filename="random_mlp_large.npz",
        grid_size=20,
        param_range=(-0.3, 0.3),
    )


def generate_optimization_trajectories(
    cache_dir: str,
    num_steps: int = 100,
) -> None:
    """
    Generate sample optimization trajectories for visualization.

    Args:
        cache_dir: Directory to save trajectory data
        num_steps: Number of steps in each trajectory
    """

    os.makedirs(cache_dir, exist_ok=True)

    from src.physics import Particle
    from src.landscape import create_dummy_data_loader

    # Create a simple model and loss function
    model = SimpleMLP(input_dim=2, hidden_dim=10, output_dim=1)
    criterion = torch.nn.MSELoss()
    train_loader = create_dummy_data_loader(batch_size=4, input_dim=2, output_dim=1)

    # Initialize loss landscape to get current params as base
    landscape = LossLandscape(
        model=model,
        criterion=criterion,
        data_loader=train_loader,
    )

    # Get current parameters as starting point
    current_params = model.get_flat_params().detach()

    # Define loss function that uses the model
    def make_loss_fn(model, criterion, train_loader):
        def loss_fn(pos):
            # Set model params from position
            model.set_flat_params(pos)
            try:
                inputs, targets = next(iter(train_loader))
            except StopIteration:
                train_loader.dataset.num_samples = 4
                loader = torch.utils.data.DataLoader(
                    train_loader.dataset, batch_size=4, shuffle=True
                )
                inputs, targets = next(iter(loader))
            outputs = model(inputs)
            return criterion(outputs, targets)
        return loss_fn

    loss_fn = make_loss_fn(model, criterion, train_loader)

    # Generate trajectories with different optimizers
    trajectories = {}

    for optimizer in ["sgd", "momentum", "adam"]:
        particle = Particle(
            initial_position=current_params.clone(),
            optimizer=optimizer,
            learning_rate=0.01,
            momentum=0.9 if optimizer == "momentum" else 0.9,
            friction=0.1 if optimizer == "momentum" else 0.1,
        )

        positions = []
        losses = []

        for _ in range(num_steps):
            pos = particle.update_position(loss_fn)
            positions.append(pos.detach().numpy())
            loss = loss_fn(pos).item()
            losses.append(loss)

        trajectories[optimizer] = {
            "positions": np.array(positions),
            "losses": np.array(losses),
        }

    # Save trajectories
    output_path = os.path.join(cache_dir, "optimization_trajectories.npz")
    np.savez_compressed(output_path, **trajectories)
    print(f"Saved optimization trajectories to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate pre-computed loss landscapes"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to save cached landscapes (default: data/precomputed_landscapes)",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["sine", "polynomial", "random", "all"],
        default="all",
        help="Type of landscapes to generate",
    )

    args = parser.parse_args()

    # Determine cache directory
    if args.cache_dir:
        cache_dir = args.cache_dir
    else:
        # Get project root and construct path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        cache_dir = os.path.join(project_root, "data", "precomputed_landscapes")

    print(f"Using cache directory: {cache_dir}")

    if args.type == "all":
        generate_sine_fitting_landscapes(cache_dir)
        generate_polynomial_fitting_landscapes(cache_dir)
        generate_random_loss_landscapes(cache_dir)
    elif args.type == "sine":
        generate_sine_fitting_landscapes(cache_dir)
    elif args.type == "polynomial":
        generate_polynomial_fitting_landscapes(cache_dir)
    elif args.type == "random":
        generate_random_loss_landscapes(cache_dir)

    print("\nDone! Pre-computed landscapes are ready for use.")


if __name__ == "__main__":
    main()
