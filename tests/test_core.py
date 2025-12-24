"""
Basic tests for PyTurboFLIM.
"""

import numpy as np
import pytest


def test_import():
    """Test that all main modules can be imported."""
    import pyturboflim
    from pyturboflim import TurboFLIM, analyze, analyze_batch
    from pyturboflim.phasor import compute_phasor, compute_phasor_batch
    from pyturboflim.io import generate_synthetic_data
    from pyturboflim.visualization import plot_lifetime_map, plot_phasor
    
    assert pyturboflim.__version__ == "1.0.0"


def test_phasor_computation():
    """Test phasor coordinate calculation."""
    from pyturboflim.phasor import compute_phasor
    
    # Single exponential decay
    t = np.linspace(0, 12.5, 100)
    decay = np.exp(-t / 2.5)  # tau = 2.5 ns
    
    G, S = compute_phasor(decay, laser_period=12.5)
    
    # Should be on or near semicircle
    assert 0 < G < 1
    assert 0 < S < 0.5


def test_synthetic_data_generation():
    """Test synthetic data generator."""
    from pyturboflim.io import generate_synthetic_data
    
    X, y = generate_synthetic_data(n_samples=100, n_bins=256)
    
    assert X.shape == (100, 256)
    assert y.shape == (100, 2)
    assert np.all(y > 0)  # Lifetimes should be positive


def test_turboflim_model():
    """Test TurboFLIM model training and prediction."""
    from pyturboflim import TurboFLIM
    from pyturboflim.io import generate_synthetic_data
    
    X, y = generate_synthetic_data(n_samples=500, n_bins=128, random_state=42)
    
    model = TurboFLIM(hidden_layers=(32, 16), max_iter=50)
    model.fit(X[:400], y[:400], verbose=False)
    
    assert model.is_fitted
    
    predictions = model.predict(X[400:])
    assert predictions.shape == (100, 2)


def test_batch_phasor():
    """Test vectorized phasor computation."""
    from pyturboflim.phasor import compute_phasor_batch
    
    decays = np.random.rand(50, 100)
    G, S = compute_phasor_batch(decays)
    
    assert G.shape == (50,)
    assert S.shape == (50,)
