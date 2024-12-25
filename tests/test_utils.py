import random
import pytest
import numpy as np
import torch
from torchevent.utils import (
    set_seed,
    expand_to_3d,
    weight_clipper,
    _parse_extra_repr,
    _tensor_to_numpy,
    _convert_state_dict_to_numpy,
    to_uint8,
    plot_event_frame,
    spike2data,
)


def test_set_seed():
    # Set the random seed
    set_seed(42)

    # Check random values for reproducibility
    random_value_cpu = torch.randint(0, 100, (1,)).item()
    random_value_cuda = (
        torch.randint(0, 100, (1,), device='cuda').item() if torch.cuda.is_available() else None
    )

    # Set the seed again and regenerate the same random values
    set_seed(42)
    random_value_cpu_2 = torch.randint(0, 100, (1,)).item()
    random_value_cuda_2 = (
        torch.randint(0, 100, (1,), device='cuda').item() if torch.cuda.is_available() else None
    )

    # Assert values are reproducible
    assert random_value_cpu == random_value_cpu_2, "CPU random values are not reproducible."
    if torch.cuda.is_available():
        assert random_value_cuda == random_value_cuda_2, "CUDA random values are not reproducible."

    set_seed(42)
    # Gradient check for reproducibility
    x = torch.randn(3, 3, requires_grad=True)
    y = torch.randn(3, 3, requires_grad=False)

    # Perform a forward and backward pass
    z = (x * y).sum()
    z.backward()

    # Capture gradients
    grad_x = x.grad.clone()

    # Reset seed and recompute
    set_seed(42)
    x = torch.randn(3, 3, requires_grad=True)
    y = torch.randn(3, 3, requires_grad=False)
    z = (x * y).sum()
    z.backward()

    # Assert gradients are the same
    assert torch.allclose(grad_x, x.grad), "Gradients are not reproducible after resetting seed."

def test_expand_to_3d():
    assert expand_to_3d(5, "test") == (5, 5, 1), "Failed for single integer input."
    assert expand_to_3d((3, 4), "test") == (3, 4, 1), "Failed for 2D tuple input."
    with pytest.raises(ValueError):
        expand_to_3d((3, 4, 5), "test")


def test_weight_clipper():
    tensor = torch.randn(3, 3) * 10
    weight_clipper(tensor, clip_value=4)
    assert torch.all(tensor <= 4) and torch.all(tensor >= -4), "Weight clipping failed."


def test_parse_extra_repr():
    extra_repr = "in_channels=3, out_channels=16, kernel_size=(3, 3), stride=1"
    args, kwargs = _parse_extra_repr(extra_repr)
    assert kwargs["in_channels"] == 3, "Failed to parse `in_channels`."
    assert kwargs["kernel_size"] == (3, 3), "Failed to parse `kernel_size`."
    assert "stride" in kwargs and kwargs["stride"] == 1, "Failed to parse `stride`."


def test_tensor_to_numpy():
    tensor = torch.randn(3, 3)
    numpy_array = _tensor_to_numpy(tensor)
    assert isinstance(numpy_array, np.ndarray), "Failed to convert tensor to numpy array."
    assert np.allclose(numpy_array, tensor.cpu().numpy()), "Converted numpy array values are incorrect."


def test_convert_state_dict_to_numpy():
    state_dict = {
        "layer1.weight": torch.randn(3, 3),
        "layer1.bias": torch.zeros(3),
    }
    numpy_state_dict = _convert_state_dict_to_numpy(state_dict)
    for key, value in state_dict.items():
        assert isinstance(numpy_state_dict[key], np.ndarray), f"Failed to convert `{key}` to numpy array."


def test_to_uint8():
    data = np.random.randn(10, 10) * 10
    uint8_data = to_uint8(data)
    assert uint8_data.dtype == np.uint8, "Output is not uint8."
    assert uint8_data.max() == 255 and uint8_data.min() == 0, "Normalization failed."


def test_plot_event_frame(tmp_path):
    event_data = np.random.randn(5, 1, 32, 32)  # (n_step, ch, width, height)
    file_name = tmp_path / "test_event_frame.png"
    plot_event_frame(event_data, file_name)
    assert file_name.exists(), "Plot event frame file not saved."

    # Additional test for polarity (ch=2)
    event_data_polarity = np.random.randn(5, 2, 32, 32)
    file_name_polarity = tmp_path / "test_event_frame_polarity.png"
    plot_event_frame(event_data_polarity, file_name_polarity)
    assert file_name_polarity.exists(), "Plot event frame file with polarity not saved."
    
def test_spike2data_summed():
    spikes = torch.randint(0, 2, (4, 10, 1, 1, 5))  # Batch=4, num_class=10, n_step=5
    data = spike2data(spikes)
    assert data.shape == (4, 10), "Summed data shape mismatch"
    expected_sum = spikes.sum(dim=4).squeeze(-1).squeeze(-1)
    assert torch.allclose(data, expected_sum), "Summed spike data is incorrect"

def test_spike2data_pred():
    spikes = torch.randint(0, 2, (4, 10, 1, 1, 5))  # Batch=4, num_class=10, n_step=5
    preds = spike2data(spikes, return_pred=True)
    assert preds.shape == (4,), "Prediction shape mismatch"
    expected_preds = spikes.sum(dim=4).squeeze(-1).squeeze(-1).argmax(dim=1)
    assert torch.equal(preds, expected_preds), "Predicted class indices are incorrect"

def test_spike2data_edge_case():
    # Edge case: all zero spikes
    spikes = torch.zeros(4, 10, 1, 1, 5)  # Batch=4, num_class=10, n_step=5
    data = spike2data(spikes)
    assert torch.all(data == 0), "Summed data should be all zeros for zero spikes"

    preds = spike2data(spikes, return_pred=True)
    assert torch.all(preds == 0), "Predicted class index should default to 0 for all zero spikes"