"""
Particle Physics Simulation Module.

This module provides the physics engine for simulating particle motion on the
loss landscape, implementing various optimization algorithms (SGD, Momentum, Adam).
"""

import torch
from typing import Optional, Tuple, List


class Particle:
    """
    Simulates a particle moving on the loss landscape according to
    gradient-based optimization rules.

    The particle's position represents model parameters, and its motion
    follows optimization algorithms like SGD, Momentum, and Adam.
    """

    def __init__(
        self,
        initial_position: torch.Tensor,
        optimizer: str = "sgd",
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        friction: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        """
        Initialize the particle with initial position and optimization parameters.

        Args:
            initial_position: Initial parameter vector (1D tensor)
            optimizer: Optimization algorithm ("sgd", "momentum", "adam")
            learning_rate: Learning rate for parameter updates
            momentum: Momentum coefficient (for momentum optimizer)
            friction: Velocity damping coefficient (for momentum)
            beta1: Exponential decay rate for first moment (Adam)
            beta2: Exponential decay rate for second moment (Adam)
            epsilon: Small value to prevent division by zero (Adam)
        """
        self.position = initial_position.clone().detach()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.friction = friction
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Velocity for momentum-based methods
        self.velocity: Optional[torch.Tensor] = None

        # Adam state variables
        self.moment1: Optional[torch.Tensor] = None  # First moment estimate
        self.moment2: Optional[torch.Tensor] = None  # Second moment estimate
        self.timestep = 0

        # History for trajectory visualization
        self.history: List[torch.Tensor] = []

    def _compute_gradient(self, loss_fn) -> torch.Tensor:
        """
        Compute the gradient of the loss function with respect to position.

        Args:
            loss_fn: Function that takes position and returns loss tensor.
                    The function should accept a 1D tensor parameter and return a scalar tensor.

        Returns:
            Gradient tensor (same shape as position)
        """
        # Make sure position has grad enabled
        if self.position.grad is not None:
            self.position.grad.zero_()

        # Compute loss using the provided function
        loss = loss_fn(self.position)

        # Check for valid loss
        if loss is None:
            raise ValueError("loss_fn returned None. Make sure loss_fn returns a scalar tensor.")

        # Backpropagation to get gradient
        try:
            loss.backward()
        except Exception as e:
            raise RuntimeError(f"Backward failed: {e}. "
                             f"The loss function must return a scalar tensor that depends on the position.")

        grad = self.position.grad
        if grad is None:
            raise ValueError("Gradient is None. The loss function may not depend on the input tensor.")

        return grad

    def update_position(self, loss_fn) -> torch.Tensor:
        """
        Update particle position based on optimization algorithm.

        Args:
            loss_fn: Function that takes position and returns scalar loss

        Returns:
            Updated position tensor
        """
        self.timestep += 1

        # Compute gradient at current position
        gradient = self._compute_gradient(loss_fn)

        if self.optimizer == "sgd":
            self.position = self._update_sgd(gradient)
        elif self.optimizer == "momentum":
            self.position = self._update_momentum(gradient)
        elif self.optimizer == "adam":
            self.position = self._update_adam(gradient)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        # Record history for trajectory visualization
        self.history.append(self.position.clone().detach())

        return self.position

    def _update_sgd(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Update position using pure SGD (gradient descent).

        Update rule: x_new = x_old - lr * gradient

        Args:
            gradient: Gradient tensor at current position

        Returns:
            New position
        """
        return self.position - self.learning_rate * gradient

    def _update_momentum(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Update position using SGD with momentum.

        Velocity update: v_new = momentum * v_old - friction * gradient
        Position update: x_new = x_old + v_new

        Args:
            gradient: Gradient tensor at current position

        Returns:
            New position
        """
        # Initialize velocity if not set
        if self.velocity is None:
            self.velocity = torch.zeros_like(self.position)

        # Update velocity with momentum and friction (damping)
        self.velocity = (
            self.momentum * self.velocity
            - self.learning_rate * gradient
            - self.friction * self.velocity
        )

        # Update position
        self.position = self.position + self.velocity

        return self.position

    def _update_adam(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Update position using Adam optimizer with adaptive learning rates.

        Update rules:
            m_t = beta1 * m_{t-1} + (1 - beta1) * gradient
            v_t = beta2 * v_{t-1} + (1 - beta2) * gradient^2
            m_hat_t = m_t / (1 - beta1^t)
            v_hat_t = v_t / (1 - beta2^t)
            x_new = x_old - lr * m_hat_t / (sqrt(v_hat_t) + epsilon)

        Args:
            gradient: Gradient tensor at current position

        Returns:
            New position
        """
        # Initialize moments if not set
        if self.moment1 is None:
            self.moment1 = torch.zeros_like(self.position)
        if self.moment2 is None:
            self.moment2 = torch.zeros_like(self.position)

        # Update biased first moment estimate
        self.moment1 = self.beta1 * self.moment1 + (1 - self.beta1) * gradient

        # Update biased second moment estimate
        self.moment2 = self.beta2 * self.moment2 + (1 - self.beta2) * (gradient ** 2)

        # Compute bias-corrected first moment
        m_hat = self.moment1 / (1 - self.beta1 ** self.timestep)

        # Compute bias-corrected second moment
        v_hat = self.moment2 / (1 - self.beta2 ** self.timestep)

        # Update position with adaptive learning rate
        self.position = self.position - self.learning_rate * m_hat / (
            torch.sqrt(v_hat) + self.epsilon
        )

        return self.position

    def get_position(self) -> torch.Tensor:
        """Return current position."""
        return self.position

    def set_position(self, new_position: torch.Tensor) -> None:
        """
        Set particle position.

        Args:
            new_position: New position tensor
        """
        self.position = new_position.clone().detach()

    def get_history(self) -> List[torch.Tensor]:
        """Return history of positions."""
        return self.history

    def clear_history(self) -> None:
        """Clear position history."""
        self.history = []

    def reset(
        self,
        initial_position: Optional[torch.Tensor] = None,
        reset_optimizer_state: bool = True,
    ) -> None:
        """
        Reset particle state.

        Args:
            initial_position: New initial position (uses current if None)
            reset_optimizer_state: Whether to clear velocity and moment states
        """
        if initial_position is not None:
            self.position = initial_position.clone().detach()

        if reset_optimizer_state:
            self.velocity = None
            self.moment1 = None
            self.moment2 = None
            self.timestep = 0

        self.history.clear()


def create_particle(
    param_vector: torch.Tensor,
    optimizer: str = "sgd",
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    friction: float = 0.1,
) -> Particle:
    """
    Factory function to create a configured particle.

    Args:
        param_vector: Initial parameter vector
        optimizer: Optimization algorithm name
        learning_rate: Learning rate
        momentum: Momentum coefficient
        friction: Velocity damping

    Returns:
        Configured Particle instance
    """
    return Particle(
        initial_position=param_vector,
        optimizer=optimizer,
        learning_rate=learning_rate,
        momentum=momentum,
        friction=friction,
    )


if __name__ == "__main__":
    # Test the Particle class with a simple quadratic loss function
    print("Testing Particle class...")

    # Create test particle at position [2, 2]
    initial_pos = torch.tensor([2.0, 2.0])
    particle = create_particle(initial_pos, optimizer="sgd", learning_rate=0.1)

    # Simple quadratic loss: loss = x^2 + y^2
    def loss_fn(pos):
        return torch.sum(pos ** 2)

    print(f"Initial position: {particle.get_position()}")
    print(f"Initial gradient: {particle._compute_gradient(loss_fn)}")

    # Run a few steps of optimization
    for i in range(5):
        particle.update_position(loss_fn)
        pos = particle.get_position()
        loss = loss_fn(pos).item()
        print(f"Step {i+1}: position={pos.detach().numpy()}, loss={loss:.6f}")

    # Test momentum optimizer
    print("\n--- Testing Momentum Optimizer ---")
    particle_momentum = create_particle(
        initial_pos, optimizer="momentum", learning_rate=0.1, momentum=0.9, friction=0.1
    )
    for i in range(5):
        particle_momentum.update_position(loss_fn)
        pos = particle_momentum.get_position()
        loss = loss_fn(pos).item()
        print(f"Step {i+1}: position={pos.detach().numpy()}, loss={loss:.6f}")

    # Test Adam optimizer
    print("\n--- Testing Adam Optimizer ---")
    particle_adam = create_particle(
        initial_pos, optimizer="adam", learning_rate=0.1
    )
    for i in range(5):
        particle_adam.update_position(loss_fn)
        pos = particle_adam.get_position()
        loss = loss_fn(pos).item()
        print(f"Step {i+1}: position={pos.detach().numpy()}, loss={loss:.6f}")

    # Test trajectory history
    print("\n--- Testing History ---")
    print(f"History length: {len(particle_adam.get_history())}")
    print("First 3 positions in history:")
    for i, pos in enumerate(particle_adam.get_history()[:3]):
        print(f"  Step {i}: {pos.detach().numpy()}")

    print("\nAll tests passed!")
