#!/usr/bin/env python
"""
Main entry point for the Loss Landscape Simulator.

This script provides command-line interface for running the simulation
with configurable physics parameters.
"""

import argparse
import dataclasses
import os
import sys

from sim.controller import create_simulation, SimulationController


@dataclasses.dataclass
class Config:
    """
    Configuration class to store all physics parameters for the simulation.

    This class provides a structured way to manage simulation parameters
    and can be used to save/load configurations.
    """

    learning_rate: float = 0.01
    momentum: float = 0.9
    friction: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    optimizer: str = "sgd"
    grid_size: int = 20
    param_range_min: float = -1.0
    param_range_max: float = 1.0
    max_iterations: int = 100
    view_mode: str = "3d"
    show_vector_field: bool = False

    def to_dict(self) -> dict:
        """Return configuration as a dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create Config instance from dictionary."""
        return cls(**config_dict)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Loss Landscape Simulator - Visualize neural network optimization dynamics"
    )

    # Physics parameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for optimization (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum coefficient for momentum/Adam optimizer (default: 0.9)",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=0.1,
        help="Velocity damping coefficient for momentum (default: 0.1)",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Beta1 for Adam optimizer (default: 0.9)",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Beta2 for Adam optimizer (default: 0.999)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for Adam optimizer (default: 1e-8)",
    )

    # Optimizer selection
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "momentum", "adam"],
        help="Optimization algorithm (default: sgd)",
    )

    # Simulation parameters
    parser.add_argument(
        "--grid-size",
        type=int,
        default=20,
        help="Number of points for loss landscape grid (default: 20)",
    )
    parser.add_argument(
        "--param-range",
        type=float,
        nargs=2,
        default=[-1.0, 1.0],
        help="Parameter range for visualization (min max) (default: -1.0 1.0)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum number of optimization steps (default: 100)",
    )

    # Output options
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Save visualization to file instead of displaying",
    )
    parser.add_argument(
        "--view-mode",
        type=str,
        default="3d",
        choices=["3d", "contour"],
        help="Visualization mode: '3d' for 3D surface, 'contour' for top-down 2D contour plot (default: 3d)",
    )
    parser.add_argument(
        "--show-vector-field",
        action="store_true",
        help="Show gradient vector field on the visualization",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate arguments
    if args.learning_rate <= 0:
        print("Error: Learning rate must be positive")
        sys.exit(1)
    if not 0 <= args.momentum < 1:
        print("Error: Momentum must be in [0, 1)")
        sys.exit(1)
    if args.grid_size < 5:
        print("Error: Grid size must be at least 5")
        sys.exit(1)

    # Create Config object from parsed arguments
    config = Config(
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        friction=args.friction,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        optimizer=args.optimizer,
        grid_size=args.grid_size,
        param_range_min=args.param_range[0],
        param_range_max=args.param_range[1],
        max_iterations=args.max_iterations,
        view_mode=args.view_mode,
        show_vector_field=args.show_vector_field,
    )

    # Create simulation with Config parameters
    sim = create_simulation(
        grid_size=config.grid_size,
        param_range=(config.param_range_min, config.param_range_max),
        learning_rate=config.learning_rate,
        momentum=config.momentum,
        friction=config.friction,
        optimizer=config.optimizer,
        view_mode=config.view_mode,
        show_vector_field=config.show_vector_field,
    )

    # Run simulation
    try:
        trajectories = sim.run_simulation(
            max_iterations=args.max_iterations,
            save_path=args.save_path,
            view_mode=args.view_mode,
            show_vector_field=config.show_vector_field,
        )
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        sys.exit(0)

    if args.quiet:
        return

    # Print summary
    print("\n" + "=" * 50)
    print("SIMULATION SUMMARY")
    print("=" * 50)
    print(f"Optimizer: {args.optimizer}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Momentum: {args.momentum}" if args.optimizer != "sgd" else "")
    print(f"Friction: {args.friction}" if args.optimizer == "momentum" else "")
    if args.optimizer == "adam":
        print(f"Beta1: {args.beta1}")
        print(f"Beta2: {args.beta2}")
    print(f"Grid size: {args.grid_size}")
    print(f"Parameter range: {args.param_range}")
    print(f"View mode: {args.view_mode}")
    print(f"Iterations: {len(trajectories)}")
    final_params = sim._get_flat_params()
    print(f"Final position: {final_params.numpy()}")


if __name__ == "__main__":
    main()
