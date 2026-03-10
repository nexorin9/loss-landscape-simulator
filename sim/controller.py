"""
Simulation Controller Module.

This module integrates the loss landscape computation, particle physics,
and visualization to create a complete simulation system.
"""

import numpy as np
import torch

from sim.renderer import Renderer
from src.model import SimpleMLP
from src.landscape import LossLandscape, create_dummy_data_loader


def compute_gradient_vector_field(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader,
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    direction1: torch.Tensor,
    direction2: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the gradient vector field at grid points.

    Args:
        model: Neural network model
        criterion: Loss function
        data_loader: Data loader for getting batches
        alpha_grid: Alpha coefficient grid
        beta_grid: Beta coefficient grid
        direction1: First interpolation direction
        direction2: Second interpolation direction

    Returns:
        Tuple of (U, V) where U and V are gradient components at each grid point
    """
    current_params = model.get_flat_params().detach()
    grad_grid_u = np.zeros(alpha_grid.shape)
    grad_grid_v = np.zeros(beta_grid.shape)

    with torch.no_grad():
        # Get a batch of data
        try:
            inputs, targets = next(iter(data_loader))
        except StopIteration:
            data_loader = create_dummy_data_loader(
                batch_size=4, input_dim=2, output_dim=1
            )
            inputs, targets = next(iter(data_loader))

    # Compute loss function that depends on parameters
    def compute_loss_for_params(params_vector):
        model.set_flat_params(params_vector)
        outputs = model(inputs)
        return criterion(outputs, targets)

    # Compute gradient at each grid point
    for i in range(alpha_grid.shape[0]):
        for j in range(alpha_grid.shape[1]):
            alpha = alpha_grid[i, j]
            beta = beta_grid[i, j]

            # Get parameter vector at this grid point
            param_vec = current_params + alpha * direction1 + beta * direction2

            # Need to compute gradient with respect to this position
            # Create a copy that tracks gradients
            param_with_grad = param_vec.clone().detach().requires_grad_(True)

            # Compute loss
            model.set_flat_params(param_with_grad)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass to get gradient
            model.zero_grad()
            loss.backward()

            # Get gradient as 1D tensor
            grads = []
            for param in model.parameters():
                if param.grad is not None:
                    grads.append(param.grad.view(-1))
            grad_vec = torch.cat(grads) if grads else torch.zeros_like(current_params)

            # Project gradient onto direction1 and direction2
            grad_u = torch.dot(grad_vec, direction1).item()
            grad_v = torch.dot(grad_vec, direction2).item()

            grad_grid_u[i, j] = grad_u
            grad_grid_v[i, j] = grad_v

    return grad_grid_u, grad_grid_v


class SimulationController:
    """
    Controls the simulation loop combining loss landscape computation,
    particle physics, and rendering.
    """

    def __init__(
        self,
        grid_size: int = 20,
        param_range: tuple = (-1.0, 1.0),
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        friction: float = 0.1,
        optimizer: str = "sgd",
        view_mode: str = "3d",
    ):
        """
        Initialize the simulation controller.

        Args:
            grid_size: Number of points for loss landscape grid
            param_range: (min, max) range for parameter interpolation
            learning_rate: Learning rate for optimization
            momentum: Momentum coefficient (for momentum/Adam optimizers)
            friction: Velocity damping coefficient (for momentum)
            optimizer: Optimization algorithm ("sgd", "momentum", "adam")
        """
        self.grid_size = grid_size
        self.param_range = param_range

        # Initialize model with parameters that will track gradients
        self.model = SimpleMLP(input_dim=2, hidden_dim=10, output_dim=1)

        # Initialize data loader
        self.data_loader = create_dummy_data_loader(
            batch_size=4, input_dim=2, output_dim=1
        )

        # Initialize loss function
        self.criterion = torch.nn.MSELoss()

        # Initialize renderer for visualization
        self.renderer = Renderer(figsize=(10, 8))

        # Visualization mode: '3d' or 'contour'
        self.view_mode = view_mode

        # Simulation state
        self.running = True
        self.iteration = 0
        self.max_iterations = 100

        # Store optimizer parameters for particle updates
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.friction = friction

    def _get_flat_params(self) -> torch.Tensor:
        """Get all model parameters as a flattened 1D tensor."""
        params_list = []
        for param in self.model.parameters():
            params_list.append(param.view(-1))
        return torch.cat(params_list)

    def run_simulation(self, max_iterations: int = 100, save_path: str = None, view_mode: str = None, show_vector_field: bool = False):
        """
        Run the simulation loop.

        Args:
            max_iterations: Maximum number of optimization steps
            save_path: If provided, save final visualization to this path
            view_mode: Visualization mode ('3d' or 'contour'), overrides instance default
            show_vector_field: Whether to display gradient vector field
        """
        self.max_iterations = max_iterations
        self.running = True

        # Use instance view_mode if not specified in call
        current_view_mode = view_mode if view_mode is not None else self.view_mode

        # Compute loss landscape surface using LossLandscape class
        print("Computing loss landscape...")
        landscape = LossLandscape(
            model=self.model,
            criterion=self.criterion,
            data_loader=self.data_loader,
            cache_dir=None,
        )

        loss_surface, alpha_grid, beta_grid = landscape.compute_loss_surface(
            grid_size=self.grid_size,
            param_range=self.param_range,
        )
        print(f"Loss landscape computed: {loss_surface.shape}")

        # Convert grids to numpy for plotting
        alpha_vals = np.linspace(self.param_range[0], self.param_range[1], self.grid_size)
        beta_vals = np.linspace(self.param_range[0], self.param_range[1], self.grid_size)
        X, Y = np.meshgrid(alpha_vals, beta_vals)

        # Compute vector field if requested
        if show_vector_field:
            print("Computing gradient vector field...")
            # Get directions from the landscape computation
            current_params = self._get_flat_params().detach()
            direction1 = torch.randn_like(current_params)
            direction1 = direction1 / torch.norm(direction1)
            direction2 = torch.randn_like(current_params)
            direction2 = direction2 - torch.dot(direction2, direction1) * direction1
            direction2 = direction2 / torch.norm(direction2)

            U, V = self.compute_vector_field(X, Y, direction1, direction2)
            self.vector_field_u = U
            self.vector_field_v = V
            print(f"Vector field computed: shape {U.shape}")
        else:
            self.vector_field_u = None
            self.vector_field_v = None

        # Initialize trajectory storage
        trajectories = []

        # Get initial parameters as a 1D tensor for position tracking
        initial_params = self._get_flat_params().detach()

        print(f"\nStarting simulation ({max_iterations} iterations)...")

        for iteration in range(max_iterations):
            if not self.running:
                break

            self.iteration = iteration + 1

            # Compute loss and gradient using the model's parameters directly
            try:
                inputs, targets = next(iter(self.data_loader))
            except StopIteration:
                self.data_loader = create_dummy_data_loader(
                    batch_size=4, input_dim=2, output_dim=1
                )
                inputs, targets = next(iter(self.data_loader))

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass to compute gradients
            self.model.zero_grad()
            loss.backward()

            # Collect gradients from model parameters
            grads = []
            for param in self.model.parameters():
                grads.append(param.grad.view(-1))
            gradient = torch.cat(grads)

            # Update position (parameters) using specified optimizer
            with torch.no_grad():
                if self.optimizer_name == "sgd":
                    # SGD: x_new = x_old - lr * gradient
                    current_params = self._get_flat_params()
                    new_params = current_params - self.learning_rate * gradient
                    self._set_flat_params(new_params)

                elif self.optimizer_name == "momentum":
                    # Momentum: v = momentum * v - lr * grad, x = x + v
                    if not hasattr(self, 'velocity'):
                        self.velocity = torch.zeros_like(initial_params)
                    else:
                        # Update velocity with momentum and friction (damping)
                        self.velocity = (
                            self.momentum * self.velocity
                            - self.learning_rate * gradient
                            - self.friction * self.velocity
                        )

                    # Update position
                    current_params = self._get_flat_params()
                    new_params = current_params + self.velocity
                    self._set_flat_params(new_params)

                elif self.optimizer_name == "adam":
                    # Adam with adaptive learning rates
                    if not hasattr(self, 'moment1'):
                        self.moment1 = torch.zeros_like(initial_params)
                        self.moment2 = torch.zeros_like(initial_params)
                        self.timestep = 0

                    self.timestep += 1
                    beta1, beta2 = 0.9, 0.999
                    eps = 1e-8

                    # Update biased first moment estimate
                    self.moment1 = beta1 * self.moment1 + (1 - beta1) * gradient
                    # Update biased second moment estimate
                    self.moment2 = beta2 * self.moment2 + (1 - beta2) * (gradient ** 2)

                    # Compute bias-corrected first moment
                    m_hat = self.moment1 / (1 - beta1 ** self.timestep)
                    # Compute bias-corrected second moment
                    v_hat = self.moment2 / (1 - beta2 ** self.timestep)

                    # Update parameters
                    current_params = self._get_flat_params()
                    new_params = current_params - self.learning_rate * m_hat / (
                        torch.sqrt(v_hat) + eps
                    )
                    self._set_flat_params(new_params)

            # Store trajectory point (in parameter space, relative to initial)
            current_params = self._get_flat_params().detach()
            trajectories.append(current_params.numpy())

            # Print progress
            if iteration % 10 == 0 or iteration == max_iterations - 1:
                print(f"Iteration {iteration + 1}/{max_iterations}: Loss = {loss.item():.6f}")

        # Generate visualization
        self._generate_visualization(X, Y, loss_surface, trajectories, save_path, view_mode=current_view_mode, show_vector_field=show_vector_field)

        return trajectories

    def _set_flat_params(self, params: torch.Tensor) -> None:
        """
        Set model parameters from a flattened 1D tensor.

        Args:
            params: Flattened parameter tensor
        """
        current_offset = 0
        for param in self.model.parameters():
            num_elements = param.numel()
            new_param = params[current_offset:current_offset + num_elements].view_as(param)
            param.data.copy_(new_param.data)
            current_offset += num_elements

    def compute_vector_field(
        self,
        alpha_grid: np.ndarray,
        beta_grid: np.ndarray,
        direction1: torch.Tensor,
        direction2: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient vector field at grid points.

        Args:
            alpha_grid: Alpha coefficient grid
            beta_grid: Beta coefficient grid
            direction1: First interpolation direction
            direction2: Second interpolation direction

        Returns:
            Tuple of (U, V) where U and V are gradient components at each grid point
        """
        current_params = self.model.get_flat_params().detach()

        with torch.no_grad():
            # Get a batch of data
            try:
                inputs, targets = next(iter(self.data_loader))
            except StopIteration:
                self.data_loader = create_dummy_data_loader(
                    batch_size=4, input_dim=2, output_dim=1
                )
                inputs, targets = next(iter(self.data_loader))

        grad_grid_u = np.zeros(alpha_grid.shape)
        grad_grid_v = np.zeros(beta_grid.shape)

        # Compute gradient at each grid point
        for i in range(alpha_grid.shape[0]):
            for j in range(alpha_grid.shape[1]):
                alpha = alpha_grid[i, j]
                beta = beta_grid[i, j]

                # Get parameter vector at this grid point
                param_vec = current_params + alpha * direction1 + beta * direction2

                # Compute loss with this parameter setting
                with torch.no_grad():
                    self.model.set_flat_params(param_vec)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                # Backward pass to get gradient (need to recompute with grad enabled)
                param_with_grad = param_vec.clone().detach().requires_grad_(True)
                self.model.set_flat_params(param_with_grad)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Clear existing gradients
                self.model.zero_grad()
                loss.backward()

                # Collect gradients from model parameters
                grads = []
                for param in self.model.parameters():
                    if param.grad is not None:
                        grads.append(param.grad.view(-1))
                grad_vec = torch.cat(grads) if grads else torch.zeros_like(current_params)

                # Project gradient onto direction1 and direction2
                grad_u = torch.dot(grad_vec, direction1).item()
                grad_v = torch.dot(grad_vec, direction2).item()

                grad_grid_u[i, j] = grad_u
                grad_grid_v[i, j] = grad_v

        return grad_grid_u, grad_grid_v

    def _generate_visualization(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        trajectories: list,
        save_path: str = None,
        view_mode: str = "3d",
        show_vector_field: bool = False,
    ):
        """
        Generate and display/save the visualization.

        Args:
            X: X coordinates grid
            Y: Y coordinates grid
            Z: Loss values grid
            trajectories: List of position arrays
            save_path: If provided, save to file instead of showing
            view_mode: Visualization mode ('3d' or 'contour')
        """
        # Clear previous plot
        self.renderer.clear()

        if view_mode == "contour":
            # Plot 2D contour view
            self.renderer.plot_contour(
                X, Y, Z,
                levels=15,
                cmap="viridis",
                add_colorbar=True,
                alpha=0.7,
            )

            # Set labels for 2D plot
            self.renderer.set_title("Loss Landscape - Contour View")
            self.renderer.set_xlabel("Parameter Direction 1")
            self.renderer.set_ylabel("Parameter Direction 2")

            # Add trajectory overlay if we have points
            if trajectories:
                traj_array = np.array(trajectories)
                self.renderer.add_trajectory(
                    positions=traj_array,
                    color="red",
                    marker_size=50,
                    line_width=2.0,
                    cmap_trajectory=True,
                )

        else:
            # Plot 3D surface (default)
            self.renderer.plot_3d_surface(
                X, Y, Z,
                alpha=0.7,
                cmap="viridis",
                contour_lines=True,
                n_contours=15,
            )

            # Set labels for 3D plot
            self.renderer.set_title("Loss Landscape Optimization")
            self.renderer.set_xlabel("Parameter Direction 1")
            self.renderer.set_ylabel("Parameter Direction 2")
            self.renderer.set_zlabel("Loss")

            # Add trajectory overlay if we have points
            if trajectories:
                traj_array = np.array(trajectories)
                self.renderer.add_trajectory(
                    positions=traj_array,
                    color="red",
                    marker_size=100,
                    line_width=2.5,
                    cmap_trajectory=True,
                )

            # Add vector field if requested
            if show_vector_field and hasattr(self, 'vector_field_u'):
                U = self.vector_field_u
                V = self.vector_field_v
                if U is not None and V is not None:
                    self.renderer.plot_vector_field(X, Y, U, V, color="blue", alpha=0.5, length=0.2)

        # Save or show
        if save_path is not None:
            self.renderer.savefig(save_path)
            print(f"\nVisualization saved to: {save_path}")
        else:
            self.renderer.show()

    def add_vector_field(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
        color: str = "red",
        alpha: float = 0.6,
        length: float = 0.3,
    ):
        """
        Add a vector field overlay to the current visualization.

        Args:
            X: X coordinates grid
            Y: Y coordinates grid
            U: Vector field x-components
            V: Vector field y-components
            color: Arrow color
            alpha: Arrow transparency
            length: Maximum arrow length
        """
        self.renderer.plot_vector_field(X, Y, U, V, color=color, alpha=alpha, length=length)

    def pause(self):
        """Pause the simulation."""
        self.running = False

    def resume(self):
        """Resume the simulation."""
        self.running = True

    def reset(self, new_optimizer: str = None, new_lr: float = None):
        """
        Reset the simulation state.

        Args:
            new_optimizer: New optimizer name if changing
            new_lr: New learning rate if changing
        """
        # Update parameters if specified
        if new_optimizer is not None:
            self.optimizer_name = new_optimizer
        if new_lr is not None:
            self.learning_rate = new_lr

        # Clear optimizer state
        if hasattr(self, 'velocity'):
            del self.velocity
        if hasattr(self, 'moment1'):
            del self.moment1
            del self.moment2
            del self.timestep

        self.iteration = 0


def create_simulation(
    grid_size: int = 20,
    param_range: tuple = (-1.0, 1.0),
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    friction: float = 0.1,
    optimizer: str = "sgd",
    view_mode: str = "3d",
    show_vector_field: bool = False,
):
    """
    Factory function to create a configured simulation.

    Args:
        grid_size: Number of points for loss landscape grid
        param_range: (min, max) range for parameter interpolation
        learning_rate: Learning rate for optimization
        momentum: Momentum coefficient
        friction: Velocity damping coefficient
        optimizer: Optimization algorithm
        view_mode: Visualization mode ('3d' or 'contour')
        show_vector_field: Whether to display gradient vector field

    Returns:
        Configured SimulationController instance
    """
    sim = SimulationController(
        grid_size=grid_size,
        param_range=param_range,
        learning_rate=learning_rate,
        momentum=momentum,
        friction=friction,
        optimizer=optimizer,
        view_mode=view_mode,
    )
    sim.show_vector_field = show_vector_field
    return sim


if __name__ == "__main__":
    # Test the simulation controller
    print("Testing SimulationController...")

    # Create simulation with SGD optimizer
    sim = create_simulation(
        grid_size=15,
        param_range=(-0.5, 0.5),
        learning_rate=0.05,
        momentum=0.9,
        friction=0.1,
        optimizer="sgd",
    )

    # Run simulation for a few iterations
    trajectories = sim.run_simulation(max_iterations=20)

    print(f"\nSimulation completed!")
    final_params = sim._get_flat_params()
    print(f"Final position: {final_params.numpy()}")
    print(f"Trajectory length: {len(trajectories)}")
