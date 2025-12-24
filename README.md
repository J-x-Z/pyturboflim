# PyTurboFLIM

[![CI](https://github.com/J-x-Z/pyturboflim/actions/workflows/ci.yml/badge.svg)](https://github.com/J-x-Z/pyturboflim/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/pyturboflim.svg)](https://badge.fury.io/py/pyturboflim)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Fast physics-guided fluorescence lifetime analysis using Phasor-Fusion neural networks.**

PyTurboFLIM combines the speed of phasor analysis with the accuracy of deep learning, achieving **5.5× faster** analysis than traditional LMA fitting while maintaining **R² > 0.9** accuracy even in low-photon regimes.

## 🚀 Installation

```bash
pip install pyturboflim
```

Or install from source:
```bash
git clone https://github.com/J-x-Z/pyturboflim.git
cd pyturboflim
pip install -e .
```

## ⚡ Quick Start

```python
from pyturboflim import TurboFLIM
from pyturboflim.io import generate_synthetic_data

# Generate training data
X_train, y_train = generate_synthetic_data(n_samples=10000)

# Train model
model = TurboFLIM()
model.fit(X_train, y_train)

# Predict lifetimes
lifetimes = model.predict(my_decay_curves)
print(f"τ₁ = {lifetimes[0, 0]:.2f} ns, τ₂ = {lifetimes[0, 1]:.2f} ns")
```

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Phasor-Fusion** | Physics-embedded neural network architecture |
| **Low-Photon Robust** | R² > 0.9 at 500 photons |
| **Fast** | 18 μs/pixel (5.5× faster than LMA) |
| **Simple API** | `model.fit()` / `model.predict()` |
| **Visualization** | Built-in lifetime maps and phasor plots |

## 📊 Performance

| Photons | PyTurboFLIM R² | Speed vs LMA |
|---------|----------------|--------------|
| 500 | 0.91 | 5.5× faster |
| 1000 | 0.94 | 5.5× faster |
| 5000 | 0.96 | 5.5× faster |

## 📖 API Reference

### Core

```python
from pyturboflim import TurboFLIM, analyze, analyze_batch
```

- `TurboFLIM()` - Main model class
- `analyze(decay)` - Quick single-curve analysis
- `analyze_batch(image)` - Batch processing

### Phasor

```python
from pyturboflim.phasor import compute_phasor, compute_phasor_batch
```

### Visualization

```python
from pyturboflim.visualization import plot_lifetime_map, plot_phasor
```

## 📝 Citation

```bibtex
@software{pyturboflim2024,
  title={PyTurboFLIM: Fast Physics-Guided Fluorescence Lifetime Analysis},
  author={Zhang, Jiaxi},
  year={2024},
  url={https://github.com/J-x-Z/pyturboflim}
}
```

## 📄 License

Apache License 2.0
