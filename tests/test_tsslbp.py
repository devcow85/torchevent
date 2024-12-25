import pytest
import torch
from torchevent.tsslbp import TSSLBP

def test_tsslbp_forward():
    inputs = torch.randn(2, 3, 4, 4, 5)  # (batch_size, channels, H, W, steps)
    tau_m = 5
    tau_s = 1
    threshold = 0.5
    syn_a = torch.ones((1, 1, 1, 1, 5))

    outputs = TSSLBP.apply(inputs, tau_m, tau_s, threshold, syn_a)
    
    assert outputs.shape == inputs.shape, "Output shape should match input shape"
    assert torch.all(outputs >= 0), "Outputs should be non-negative"

def test_tsslbp_backward():
    inputs = torch.randn(2, 3, 4, 4, 5, requires_grad=True)
    tau_m = 5
    tau_s = 1
    threshold = 0.5
    syn_a = torch.ones((1, 1, 1, 1, 5))
    
    outputs = TSSLBP.apply(inputs, tau_m, tau_s, threshold, syn_a)
    grad_outputs = torch.randn_like(outputs)
    outputs.backward(grad_outputs)
    
    assert inputs.grad is not None, "Gradient should be calculated"
    assert inputs.grad.shape == inputs.shape, "Gradient shape should match input shape"

def test_tsslbp_edge_cases():
    inputs = torch.zeros(2, 3, 4, 4, 5)
    tau_m = 5
    tau_s = 1
    threshold = 0.5
    syn_a = torch.ones((1, 1, 1, 1, 5))

    outputs = TSSLBP.apply(inputs, tau_m, tau_s, threshold, syn_a)
    assert torch.all(outputs == 0), "Outputs should be zero for zero inputs"