import pytest
import torch
from torchevent.profile import LayerwiseProfiler
from torchevent.models import NCARSNet, NMNISTNet  # Import required models


@pytest.fixture
def model():
    """
    Fixture to create a test model.
    You can switch between NCARSNet and NMNISTNet for testing.
    """
    return NMNISTNet(tau_m=5, tau_s=1, n_steps=5)  # Use NMNISTNet


@pytest.fixture
def profiler(model):
    """Fixture to create a profiler for the test model."""
    return LayerwiseProfiler(model)


def test_profiler_trace(profiler, model):
    """Test trace profiling."""
    input_tensor = torch.rand((1, 5, 2, 34, 34))  # Input tensor
    with profiler.profile(profile_type='trace') as prof:
        _ = model(input_tensor)

    data = profiler.get_data()

    # Assertions
    assert len(data) > 0, "Profiler should collect data for all layers"
    assert data[0]["module_type"] == "SNNConv3d", "First layer should be SNNConv3d"
    assert "SNNLinear" in [layer["module_type"] for layer in data], "Linear layer should be present in the model"


def test_profiler_summary(profiler, model):
    """Test summary profiling."""
    input_tensor = torch.rand((1, 5, 2, 34, 34))  # Input tensor
    with profiler.profile(profile_type='summary') as prof:
        _ = model(input_tensor)

    data = profiler.get_data()

    # Assertions
    assert len(data) > 0, "Profiler should collect data for all layers"
    assert "Params" in data[0], "Summary data should include Params"
    assert "Latency" in data[0], "Summary data should include Latency"


def test_profiler_pickle(profiler, model, tmp_path):
    """Test pickle saving functionality."""
    input_tensor = torch.rand((1, 5, 2, 34, 34))  # Input tensor
    with profiler.profile(profile_type='trace') as prof:
        _ = model(input_tensor)

    pickle_path = tmp_path / "test_profiler.pkl"
    profiler.to_pickle(pickle_path)

    # Assertions
    assert pickle_path.exists(), "Pickle file should be created"
    assert len(profiler.get_data()) > 0, "Profiler should collect data for all layers"


def test_profiler_clear(profiler, model):
    """Test profiler's clear functionality."""
    input_tensor = torch.rand((1, 5, 2, 34, 34))  # Input tensor
    with profiler.profile(profile_type='summary') as prof:
        _ = model(input_tensor)

    profiler.clear()
    data = profiler.original_forwards

    # Assertions
    assert len(data) == 0, "Profiler data should be cleared"
