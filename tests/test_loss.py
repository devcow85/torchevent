import pytest
import torch
from torchevent.loss import SpikeKernelLoss, SpikeCountLoss, SpikeSoftmaxLoss

def test_spike_kernel_loss():
    outputs = torch.randn(2, 3, 1, 1, 10)  # Batch=2, num_classes=3, n_steps=10
    target = torch.randint(0, 2, (2, 3, 1, 1, 10)).float()

    loss_fn = SpikeKernelLoss(tau_s=5.0)
    loss = loss_fn(outputs, target)

    assert loss.item() > 0, "Loss should be positive"
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"

def test_spike_count_loss():
    outputs = torch.randint(0, 2, (2, 3, 1, 1, 10)).float()  # Batch=2, num_classes=3, n_steps=10
    labels = torch.tensor([0, 2])

    loss_fn = SpikeCountLoss(desired_count=5, undesired_count=1)
    loss = loss_fn(outputs, labels)

    assert loss.item() > 0, "Loss should be positive"
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"

def test_spike_softmax_loss():
    outputs = torch.randn(2, 3, 1, 1, 10)  # Batch=2, num_classes=3, n_steps=10
    target = torch.tensor([0, 2])

    loss_fn = SpikeSoftmaxLoss()
    loss = loss_fn(outputs, target)

    assert loss.item() > 0, "Loss should be positive"
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"

def test_spike_kernel_loss_psp():
    inputs = torch.randint(0, 2, (2, 3, 1, 1, 10)).float()
    loss_fn = SpikeKernelLoss(tau_s=5.0)
    psp_output = loss_fn._psp(inputs, tau_s=5.0)

    assert psp_output.shape == inputs.shape, "PSP output shape mismatch"
    assert torch.all(psp_output >= 0), "PSP values should be non-negative"