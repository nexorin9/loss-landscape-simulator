# Loss Landscape Simulator

An interactive physics simulation that visualizes neural network optimization dynamics by modeling the loss landscape as a physical surface where particles move according to gradient-based forces. This project combines numerical methods, visualization, and educational exploration of deep learning optimization.

## Overview

The loss landscape of a neural network is a complex, high-dimensional surface that represents the model's error across different weight configurations. Understanding this landscape is crucial for comprehending how optimization algorithms like SGD, Momentum, and Adam work.

This simulator:
- Visualizes 2D/3D slices of loss landscapes
- Simulates particle motion using gradient-based forces
- Supports multiple optimization algorithms (SGD, Momentum, Adam)
- Shows trajectories and gradient vector fields

## Installation

```bash
pip install torch numpy scipy matplotlib
```

## Usage

Run the main simulation:

```bash
python sim/main.py
```

### Command-line Arguments

- `--learning-rate` or `-lr`: Learning rate for optimizer (default: 0.01)
- `--momentum`: Momentum coefficient (default: 0.9)
- `--friction`: Velocity damping factor (default: 0.1)
- `--optimizer`: Optimization algorithm (sgd, momentum, adam; default: sgd)
- `--view-mode`: Visualization mode (3d, contour; default: 3d)
- `--grid-size`: Grid resolution for landscape (default: 25)

### Examples

```bash
# Basic SGD with default parameters
python sim/main.py

# Momentum optimizer with higher learning rate
python sim/main.py --optimizer momentum --learning-rate 0.1

# Adam optimizer with custom friction
python sim/main.py --optimizer adam --friction 0.05

# Contour view instead of 3D surface
python sim/main.py --view-mode contour
```

## Pre-computed Landscapes

The simulator includes pre-computed loss landscapes for various models. These can be loaded quickly without recomputing:

```bash
# Generate new pre-computed landscapes
python data/generate_landscapes.py --type all

# Types: sine, polynomial, random, or all
```

Pre-computed landscapes are stored in `data/precomputed_landscapes/`:
- `sine_mlp_*.npz`: Loss landscapes for sine curve fitting with different model sizes
- `polynomial_mlp.npz`: Polynomial function fitting landscape
- `random_mlp_*.npz`: Randomly initialized MLP landscapes

## How It Works

## How It Works

### Loss Landscape

The loss landscape represents the neural network's loss (error) as a function of its weights. For a simple MLP with one hidden layer, we can visualize a 2D slice of this landscape by interpolating between two points in weight space.

### Optimization Algorithms

**SGD (Stochastic Gradient Descent)**:
```
x_new = x_old - lr * gradient
```

**Momentum**:
```
velocity = momentum * velocity + gradient
x_new = x_old - lr * velocity
```

**Adam**:
Uses adaptive learning rates with momentum, maintaining running averages of gradients and their squares.

## Physics Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| Learning Rate | Step size for parameter updates | 0.001 - 0.1 |
| Momentum | Velocity retention factor (SGD with momentum) | 0.5 - 0.99 |
| Friction | Velocity damping coefficient | 0.01 - 0.5 |

### Optimizer-Specific Parameters

**Adam Optimizer**:
- `beta1`: First moment decay rate (default: 0.9)
- `beta2`: Second moment decay rate (default: 0.999)
- `epsilon`: Numerical stability constant (default: 1e-8)

## Advanced Usage

### Custom Learning Rates

```bash
# Aggressive optimization with high learning rate
python sim/main.py --learning-rate 0.1 --friction 0.3

# Conservative optimization with low learning rate
python sim/main.py --learning-rate 0.001 --friction 0.01
```

### Comparing Optimizers

```bash
# Compare SGD vs Momentum on the same landscape
python sim/main.py --optimizer sgd --view-mode contour
python sim/main.py --optimizer momentum --view-mode contour
```

## Troubleshooting

**Issue**: Simulation runs but shows a flat surface.

**Solution**: This may indicate numerical issues. Try adjusting the learning rate or parameter range. A very small learning rate may cause slow convergence that appears as a flat surface.

**Issue**: Loss values are NaN or infinite.

**Solution**: Reduce the learning rate. High learning rates can cause optimization to diverge.

## Project Structure

```
loss-landscape-simulator/
├── src/              # Core modules
│   ├── model.py      # Neural network definitions (MLP, CNN)
│   ├── landscape.py  # Loss surface computation
│   └── physics.py    # Particle physics simulation (SGD, Momentum, Adam)
├── sim/              # Simulation modules
│   ├── renderer.py   # Visualization engine (3D, contour, vector field)
│   ├── controller.py # Main simulation loop
│   └── main.py       # Entry point
└── data/
    ├── datasets/
    │   └── synthetic.py  # Demo datasets (sine, polynomial fitting)
    └── precomputed_landscapes/  # Pre-computed loss landscapes (.npz files)
```

## Development

### Running Tests

```bash
# Test model definitions
python src/model.py

# Test landscape computation
python src/landscape.py

# Test physics simulation
python src/physics.py
```

### Adding New Features

1. Modify `src/model.py` to add new network architectures
2. Update `src/physics.py` for new optimization algorithms
3. Extend `sim/renderer.py` for additional visualization modes
4. Add datasets in `data/datasets/`

## Dependencies

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- Matplotlib

---

## 支持作者

如果您觉得这个项目对您有帮助，欢迎打赏支持！

![Buy Me a Coffee](buymeacoffee.png)

**Buy me a coffee (crypto)**

| 币种 | 地址 |
|------|------|
| BTC | `bc1qc0f5tv577yt59tw8sqaq3tey98xehy32frzd` |
| ETH / USDT | `0x3b7b6c47491e4778157f0756102f134d05070704` |
| SOL | `6Xuk373zc6x6XWcAAuqvbWW92zabJdCmN3CSwpsVM6sd` |
