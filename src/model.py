"""
Neural Network Models for Loss Landscape Visualization.

This module provides simple neural network architectures suitable for
visualizing loss landscape dynamics in deep learning optimization.
"""

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    A simple 1-hidden-layer MLP for loss landscape visualization.

    Architecture: input -> hidden (10 units) -> output
    Uses tanh activation for smooth gradient behavior.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 10, output_dim: int = 1):
        """
        Initialize the MLP.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Number of units in hidden layer
            output_dim: Dimension of output (default: 1 for regression)
        """
        super(SimpleMLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Layer definitions
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def get_flat_params(self) -> torch.Tensor:
        """
        Return all parameters as a 1D tensor.

        Returns:
            Flattened parameters tensor
        """
        params_list = []
        for param in self.parameters():
            params_list.append(param.view(-1))
        return torch.cat(params_list)

    def set_flat_params(self, params: torch.Tensor) -> None:
        """
        Load parameters from a 1D tensor.

        Args:
            params: Flattened parameters tensor
        """
        current_offset = 0
        for param in self.parameters():
            num_elements = param.numel()
            new_param = params[current_offset:current_offset + num_elements].view_as(param)
            param.data.copy_(new_param.data)
            current_offset += num_elements


class SmallCNN(nn.Module):
    """
    A small CNN for comparison with MLP.

    Architecture:
        Input -> Conv2d(6) -> ReLU -> MaxPool2d -> Conv2d(16) -> ReLU -> MaxPool2d
        -> Flatten -> Linear(120) -> ReLU -> Linear(84) -> ReLU -> Linear(num_classes)
    """

    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        """
        Initialize the CNN.

        Args:
            input_channels: Number of input channels (default: 1 for grayscale)
            num_classes: Number of output classes (default: 10 for CIFAR/MNIST)
        """
        super(SmallCNN, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate flatten size (assuming input is 32x32)
        # Calculate flatten size by forward passing a dummy tensor
        dummy_input = torch.randn(1, input_channels, 32, 32)
        self._conv_output_size = self.features(dummy_input).numel()

        self.classifier = nn.Sequential(
            nn.Linear(self._conv_output_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_channels, height, width)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_flat_params(self) -> torch.Tensor:
        """Return all parameters as a 1D tensor."""
        params_list = []
        for param in self.parameters():
            params_list.append(param.view(-1))
        return torch.cat(params_list)

    def set_flat_params(self, params: torch.Tensor) -> None:
        """Load parameters from a 1D tensor."""
        current_offset = 0
        for param in self.parameters():
            num_elements = param.numel()
            new_param = params[current_offset:current_offset + num_elements].view_as(param)
            param.data.copy_(new_param.data)
            current_offset += num_elements


def test_models():
    """Test that models can be instantiated and forward pass works."""
    # Test MLP
    mlp = SimpleMLP(input_dim=2, hidden_dim=10, output_dim=1)
    x_mlp = torch.randn(5, 2)
    output_mlp = mlp(x_mlp)
    assert output_mlp.shape == (5, 1), f"MLP output shape error: {output_mlp.shape}"

    # Test get_flat_params
    flat_params = mlp.get_flat_params()
    assert flat_params.dim() == 1, "Flat params should be 1D"

    # Test set_flat_params
    new_flat_params = torch.randn_like(flat_params)
    mlp.set_flat_params(new_flat_params)
    flat_params_after = mlp.get_flat_params()
    assert torch.allclose(flat_params_after, new_flat_params, atol=1e-6), "set_flat_params failed"

    # Test CNN
    cnn = SmallCNN(input_channels=1, num_classes=10)
    x_cnn = torch.randn(5, 1, 32, 32)
    output_cnn = cnn(x_cnn)
    assert output_cnn.shape == (5, 10), f"CNN output shape error: {output_cnn.shape}"

    # Test CNN get_flat_params
    flat_params_cnn = cnn.get_flat_params()
    assert flat_params_cnn.dim() == 1, "CNN Flat params should be 1D"

    print("All model tests passed!")


if __name__ == "__main__":
    test_models()
