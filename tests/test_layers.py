import pytest
import torch
from torchevent.layers import SNNConv3d, SNNLinear, SNNDropout, SNNSumPooling

def test_snnconv3d_forward():
    layer = SNNConv3d(3, 16, kernel_size=3, stride=1, padding=1, tau_m=5, tau_s=1, threshold=1.0, n_steps=5)
    inputs = torch.randn(2, 3, 32, 32, 5)  # (batch_size, channels, H, W, steps)
    outputs = layer(inputs)
    
    assert outputs.shape == (2, 16, 32, 32, 5), "Output shape mismatch"

def test_snnlinear_forward():
    layer = SNNLinear(64, 10, tau_m=5, tau_s=1, threshold=1.0, n_steps=5)
    inputs = torch.randn(2, 64, 1, 1, 5)  # (batch_size, in_features, 1, 1, steps)
    outputs = layer(inputs)
    
    assert outputs.shape == (2, 10, 1, 1, 5), "Output shape mismatch"

def test_snndropout():
    layer = SNNDropout(p=0.5)
    inputs = torch.randn(2, 3, 32, 32, 5)  # (batch_size, channels, H, W, steps)
    outputs = layer(inputs)
    
    assert outputs.shape == inputs.shape, "Dropout should not change shape"

def test_snnsumpooling():
    layer = SNNSumPooling(kernel_size=2, stride=2)
    inputs = torch.randn(2, 3, 32, 32, 5)  # (batch_size, channels, H, W, steps)
    outputs = layer(inputs)
    
    assert outputs.shape == (2, 3, 16, 16, 5), "Output shape mismatch"

def test_weight_clipping():
    layer = SNNConv3d(3, 16, kernel_size=3, stride=1, padding=1)
    layer.weight_clipper(clip_value=2)
    assert torch.all(layer.weight <= 2) and torch.all(layer.weight >= -2), "Weight clipping failed"