import pickle

import torch
import torch.nn as nn

import torchevent.layers as L
from torchevent.utils import _convert_state_dict_to_numpy, _parse_extra_repr, _tensor_to_numpy

def conv_pool_block(in_channels, out_channels, kernel_size, padding, pooling_size, pooling_stride, tsslbp_config):
    """
    A block combining convolution and pooling layers with TSSLBP configuration.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        padding (int): Padding for the convolutional layer.
        pooling_size (int): Pooling kernel size.
        pooling_stride (int): Pooling stride.
        tsslbp_config (dict): Configuration for TSSLBP.
    
    Returns:
        nn.Sequential: A sequential block of convolution and pooling layers.
    """
    return nn.Sequential(
        L.SNNConv3d(in_channels, out_channels, kernel_size, padding=padding, **tsslbp_config),
        L.SNNSumPooling(pooling_size, pooling_stride)
    )

# Base Network Class
class BaseNet(nn.Module):
    def __init__(self, tau_m, tau_s, n_steps):
        super(BaseNet, self).__init__()
        self.tsslbp_config = {
            'use_tsslbp': True,
            'tau_m': tau_m,
            'tau_s': tau_s,
            'n_steps': n_steps
        }
        self.layers = nn.ModuleList()  # A single ModuleList to store all layers (conv and fc)
        
        self.trace_dict = {}
        self.hooks = []  # Store forward hooks


    # Helper method to create layers dynamically
    def _make_layers(self, layer_configs):
        layers = []
        for layer_class, layer_params in layer_configs:
            layer = layer_class(**layer_params)
            layers.append(layer)
        return layers

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)  # Assuming the permutation is common across models

        for layer in self.layers:
            x = layer(x)

        return x
    
    def _hook_fn(self, name):
        
        def hook(module, inputs, outputs):
            self.trace_dict[name] = {
                "module_type": type(module).__name__,
                "inputs": tuple(_tensor_to_numpy(inp) for inp in inputs),
                "outputs": _tensor_to_numpy(outputs),
                "extra_repr": _parse_extra_repr(module.extra_repr()),
                "state_dict": _convert_state_dict_to_numpy(module.state_dict())
                }
        return hook
        
    def register_hook(self):
        if self.hooks:
            print("Hooks are already registered.")
            return
        
        for name, module in self.named_modules():
            # Check if the module has no submodules (leaf module)
            if len(list(module.children())) == 0:
                hook = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)

    def remove_hook(self):
        for hook in self.hooks:
            try:
                hook.remove()
            except Exception as e:
                print(f"Failed to remove hook: {e}")
        self.hooks.clear()
    
    def weight_clipper(self):
        for name, module in self.named_modules():
            # Check if the module has no submodules (leaf module)
            if len(list(module.children())) == 0:
                if hasattr(module, 'weight_clipper'):
                    module.weight_clipper()
                    
    def save_trace_dict(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.trace_dict, f)
            
    def save_model(self, filename):
        model_data = {
            "state_dict": self.state_dict(),
            "tsslbp_config": self.tsslbp_config
        }
        torch.save(model_data, filename)
        print(f"Model save to {filename}")
        
    def load_model(self, filename):
        model_data = torch.load(filename)
        
        keys_to_delete = [key for key in model_data['state_dict'].keys() if 'membrain_input' in key]
        for key in keys_to_delete:
            del model_data['state_dict'][key]
        
        self.load_state_dict(model_data['state_dict'])
        self.tsslbp_config = model_data['tsslbp_config']
        
        print(f"Model loaded from {filename}")
        print("tsslbp config changed!")
        print(self.tsslbp_config)
        
    def summary(self, input_shape):
        self.register_hook()
        self.forward(torch.rand((1,)+input_shape))
        self.remove_hook()
        
        key_data = list(self.trace_dict.keys())
        
        
        summary_dict = {
            "num_layers": len(key_data),
            "input_shape": input_shape,
            "num_classes": self.trace_dict[key_data[-1]]['outputs'].shape[1],
            "num_parameters": sum(p.numel() for p in self.parameters())
        }
        
        self.trace_dict = {}
        
        return summary_dict | self.tsslbp_config

# NCARS Network 64x64 input
class NCARSNet(BaseNet):
    def __init__(self, tau_m, tau_s, n_steps, weight = None):
        super(NCARSNet, self).__init__(tau_m, tau_s, n_steps)
        
        # Configuration of layers for NCARSNet
        layer_configs = [
            (conv_pool_block, {'in_channels': 1, 'out_channels': 15, 'kernel_size': 5, 'padding': 2, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}),
            (conv_pool_block, {'in_channels': 15, 'out_channels': 40, 'kernel_size': 5, 'padding': 2, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}),
            (conv_pool_block, {'in_channels': 40, 'out_channels': 80, 'kernel_size': 3, 'padding': 1, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}),
            (conv_pool_block, {'in_channels': 80, 'out_channels': 160, 'kernel_size': 3, 'padding': 1, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}),
            (conv_pool_block, {'in_channels': 160, 'out_channels': 320, 'kernel_size': 3, 'padding': 1, 'pooling_size': 4, 'pooling_stride': 4, 'tsslbp_config': self.tsslbp_config}),
            (L.SNNLinear, {'in_features': 320, 'out_features': 64, **self.tsslbp_config}),
            (L.SNNLinear, {'in_features': 64, 'out_features': 2, **self.tsslbp_config}),
        ]

        # Create the layers dynamically and store them in self.layers
        self.layers = nn.ModuleList(self._make_layers(layer_configs))
        
        # If a weight path is provided, load the model weights
        if weight is not None:
            self.load_model(weight)

# NMNIST Network 34x34 input
class NMNISTNet(BaseNet):
    def __init__(self, tau_m, tau_s, n_steps, weight = None):
        super(NMNISTNet, self).__init__(tau_m, tau_s, n_steps)
        
        # Configuration of layers for NMNISTNet
        layer_configs = [
            (conv_pool_block, {'in_channels': 2, 'out_channels': 12, 'kernel_size': 5, 'padding': 1, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}),
            (conv_pool_block, {'in_channels': 12, 'out_channels': 64, 'kernel_size': 5, 'padding': 0, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}),
            (L.SNNLinear, {'in_features': 2304, 'out_features': 10, **self.tsslbp_config}),
        ]

        # Create the layers dynamically and store them in self.layers
        self.layers = nn.ModuleList(self._make_layers(layer_configs))
        
        # If a weight path is provided, load the model weights
        if weight is not None:
            self.load_model(weight)

# DVSGesture Network 128x128 input
class DVSGestureNet(BaseNet):
    def __init__(self, tau_m, tau_s, n_steps, weight = None):
        super(DVSGestureNet, self).__init__(tau_m, tau_s, n_steps)
        
        # Configuration of layers for DVSGestureNet
        layer_configs = [
            (conv_pool_block, {'in_channels': 2, 'out_channels': 15, 'kernel_size': 5, 'padding': 2, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}), 
            (conv_pool_block, {'in_channels': 15, 'out_channels': 40, 'kernel_size': 5, 'padding': 2, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}), 
            (conv_pool_block, {'in_channels': 40, 'out_channels': 80, 'kernel_size': 3, 'padding': 1, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}), 
            (conv_pool_block, {'in_channels': 80, 'out_channels': 160, 'kernel_size': 3, 'padding': 1, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}), 
            (conv_pool_block, {'in_channels': 160, 'out_channels': 320, 'kernel_size': 3, 'padding': 1, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}), 
            (L.SNNLinear, {'in_features': 5120, 'out_features': 512, **self.tsslbp_config}),
            (L.SNNLinear, {'in_features': 512, 'out_features': 11, **self.tsslbp_config}),
        ]

        # Create the layers dynamically and store them in self.layers
        self.layers = nn.ModuleList(self._make_layers(layer_configs))
        
        # If a weight path is provided, load the model weights
        if weight is not None:
            self.load_model(weight)
            

# DVSGesture Network 64x64 input
class PPGen4NetMini(BaseNet):
    def __init__(self, tau_m, tau_s, n_steps, weight = None):
        super(PPGen4NetMini, self).__init__(tau_m, tau_s, n_steps)
        
        # Configuration of layers for NCARSNet
        layer_configs = [
            (conv_pool_block, {'in_channels': 1, 'out_channels': 15, 'kernel_size': 5, 'padding': 2, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}), #32
            (conv_pool_block, {'in_channels': 15, 'out_channels': 40, 'kernel_size': 5, 'padding': 2, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}), #16
            (conv_pool_block, {'in_channels': 40, 'out_channels': 80, 'kernel_size': 3, 'padding': 1, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}), #8
            (L.SNNConv3d, {'in_channels': 80, 'out_channels': 160, 'kernel_size': 3, 'padding': 1, 'stride': 2, **self.tsslbp_config}), # 4
            (conv_pool_block, {'in_channels': 160, 'out_channels': 320, 'kernel_size': 3, 'padding': 1, 'pooling_size': 4, 'pooling_stride': 4, 'tsslbp_config': self.tsslbp_config}), # ?
            (L.SNNLinear, {'in_features': 320, 'out_features': 64, **self.tsslbp_config}),
            (L.SNNLinear, {'in_features': 64, 'out_features': 5, **self.tsslbp_config}),
        ]

        # Create the layers dynamically and store them in self.layers
        self.layers = nn.ModuleList(self._make_layers(layer_configs))
        
        # If a weight path is provided, load the model weights
        if weight is not None:
            self.load_model(weight)

# DVSGesture Network 64x64 input
class PPGen4NetMini128(BaseNet):
    def __init__(self, tau_m, tau_s, n_steps, weight = None):
        super(PPGen4NetMini128, self).__init__(tau_m, tau_s, n_steps)
        
        # Configuration of layers for NCARSNet
        layer_configs = [
            (conv_pool_block, {'in_channels': 1, 'out_channels': 15, 'kernel_size': 5, 'padding': 2, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}), # 40
            (conv_pool_block, {'in_channels': 15, 'out_channels': 40, 'kernel_size': 5, 'padding': 2, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}), # 20
            (conv_pool_block, {'in_channels': 40, 'out_channels': 80, 'kernel_size': 3, 'padding': 1, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}), # 10
            (L.SNNConv3d, {'in_channels': 80, 'out_channels': 160, 'kernel_size': 3, 'padding': 1, 'stride': 1, **self.tsslbp_config}), # 10
            (conv_pool_block, {'in_channels': 160, 'out_channels': 320, 'kernel_size': 3, 'padding': 1, 'pooling_size': 10, 'pooling_stride': 10, 'tsslbp_config': self.tsslbp_config}),  # 5 -> 1
            (L.SNNLinear, {'in_features': 320, 'out_features': 64, **self.tsslbp_config}),
            (L.SNNLinear, {'in_features': 64, 'out_features': 5, **self.tsslbp_config}),
        ]

        # Create the layers dynamically and store them in self.layers
        self.layers = nn.ModuleList(self._make_layers(layer_configs))
        
        # If a weight path is provided, load the model weights
        if weight is not None:
            self.load_model(weight)