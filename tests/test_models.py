import pytest
import torch
from torchevent.models import NCARSNet, NMNISTNet, DVSGestureNet, PGen4NetMini

def test_ncarsnet_forward():
    model = NCARSNet(tau_m=5, tau_s=1, n_steps=5)
    inputs = torch.randn(2, 5, 1, 64, 64)  # (batch_size, steps, channels, H, W)
    outputs = model(inputs)
    
    assert outputs.shape == (2, 2, 1, 1, 5), "Output shape mismatch for NCARSNet"

def test_nmnistnet_forward():
    model = NMNISTNet(tau_m=5, tau_s=1, n_steps=5)
    inputs = torch.randn(2, 5, 2, 34, 34)  # (batch_size, steps, channels, H, W)
    outputs = model(inputs)
    
    assert outputs.shape == (2, 10, 1, 1, 5), "Output shape mismatch for NMNISTNet"

def test_dvsgesturenet_forward():
    model = DVSGestureNet(tau_m=5, tau_s=1, n_steps=5)
    inputs = torch.randn(2, 5, 2, 128, 128)  # (batch_size, steps, channels, H, W)
    outputs = model(inputs)
    
    assert outputs.shape == (2, 11, 1, 1, 5), "Output shape mismatch for DVSGestureNet"

def test_ppgen4netmini_forward():
    model = PGen4NetMini(tau_m=5, tau_s=1, n_steps=5)
    inputs = torch.randn(2, 5, 1, 64, 64)  # (batch_size, steps, channels, H, W)
    outputs = model(inputs)
    
    assert outputs.shape == (2, 5, 1, 1, 5), "Output shape mismatch for PPGen4NetMini"

def test_model_weight_save_load(tmp_path):
    model = NCARSNet(tau_m=5, tau_s=1, n_steps=5)
    filepath = tmp_path / "model.pth"
    
    model.save_model(filepath)
    assert filepath.exists(), "Model file not saved"
    
    loaded_model = NCARSNet(tau_m=5, tau_s=1, n_steps=5)
    loaded_model.load_model(filepath)
    assert model.tsslbp_config == loaded_model.tsslbp_config, "TSSLBP config mismatch after loading"