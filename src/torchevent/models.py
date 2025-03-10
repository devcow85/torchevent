import pandas as pd

import torch
import torch.nn as nn

import torchevent.layers as L
from torchevent.profile import LayerwiseProfiler

def conv_pool_block(in_channels, out_channels, kernel_size, padding, pooling_size, pooling_stride, tsslbp_config):
    return nn.Sequential(
        L.SNNConv3d(in_channels, out_channels, kernel_size, padding=padding, **tsslbp_config),
        L.SNNSumPooling(pooling_size, pooling_stride)
    )

class BaseNet(nn.Module):
    def __init__(self, tau_m, tau_s, n_steps):
        super(BaseNet, self).__init__()
        self.tsslbp_config = {
            'use_tsslbp': True,
            'tau_m': tau_m,
            'tau_s': tau_s,
            'n_steps': n_steps
        }
        self.layers = nn.ModuleList()
        
        self.profiler = LayerwiseProfiler(self)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        for layer in self.layers:
            x = layer(x)
        return x

    def _make_layers(self, layer_configs):
        layers = []
        for layer_class, layer_params in layer_configs:
            layer = layer_class(**layer_params)
            layers.append(layer)
        return layers

    def weight_clipper(self):
        for _, module in self.named_modules():
            if len(list(module.children())) == 0:
                if hasattr(module, 'weight_clipper'):
                    module.weight_clipper()
                
    def save_model(self, filename = None, return_dict = False):
        model_data = {
            "state_dict": self.state_dict(),
            "tsslbp_config": self.tsslbp_config
        }
        if return_dict:
            return model_data
        
        torch.save(model_data, filename)
        print(f"Model save to {filename}")
        
        
    def load_model(self, filename = None, model_data = None):
        if filename is not None:
            model_data = torch.load(filename)
        state_dict = model_data['state_dict']

        keys_to_delete = []
        for key in state_dict.keys():
            parts = key.split('.')
            module = self
            exists = True

            for part in parts[:-1]:
                if not hasattr(module, part):
                    exists = False
                    break
                module = getattr(module, part)

            if not exists or not hasattr(module, parts[-1]):
                keys_to_delete.append(key)

        for key in keys_to_delete:
            print(f"Deleting key: {key} from state_dict")
            del state_dict[key]

        self.load_state_dict(state_dict, strict=False)
        self.tsslbp_config = model_data.get('tsslbp_config', None)
        print(f"Model loaded from {filename}")
        print("tsslbp config updated:", self.tsslbp_config)

    
    def trace(self, input_data):
        if isinstance(input_data, tuple):
            dummy_input = torch.rand((1,) + input_data)
        elif isinstance(input_data, torch.Tensor):
            dummy_input = input_data
        else:
            raise ValueError("input_data must be either a tuple (input shape) or a torch.Tensor")

        with self.profiler.profile(profile_type='trace') as prof:
            self(dummy_input) 

        return self.profiler.get_data()
    
    def summary(self, input_shape):
        dummy_input = torch.rand((1,) + input_shape)
        with self.profiler.profile(profile_type='summary') as prof:
            _ = self(dummy_input)

        summary_data = self.profiler.get_data()
        
        df = pd.DataFrame(summary_data)
        
        print(df.to_string())

        return df

# NCARS Network 64x64 input
class NCARSNet(BaseNet):
    def __init__(self, tau_m, tau_s, n_steps, weight = None):
        super(NCARSNet, self).__init__(tau_m, tau_s, n_steps)
        
        layer_configs = [
            (conv_pool_block, {'in_channels': 1, 'out_channels': 15, 'kernel_size': 5, 'padding': 2, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}),
            (conv_pool_block, {'in_channels': 15, 'out_channels': 40, 'kernel_size': 5, 'padding': 2, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}),
            (conv_pool_block, {'in_channels': 40, 'out_channels': 80, 'kernel_size': 3, 'padding': 1, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}),
            (conv_pool_block, {'in_channels': 80, 'out_channels': 160, 'kernel_size': 3, 'padding': 1, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}),
            (conv_pool_block, {'in_channels': 160, 'out_channels': 320, 'kernel_size': 3, 'padding': 1, 'pooling_size': 4, 'pooling_stride': 4, 'tsslbp_config': self.tsslbp_config}),
            (L.SNNLinear, {'in_features': 320, 'out_features': 64, **self.tsslbp_config}),
            (L.SNNLinear, {'in_features': 64, 'out_features': 2, **self.tsslbp_config}),
        ]

        self.layers = nn.ModuleList(self._make_layers(layer_configs))
        
        if weight is not None:
            self.load_model(weight)

# NMNIST Network 34x34 input
class NMNISTNet(BaseNet):
    def __init__(self, tau_m, tau_s, n_steps, weight = None):
        super(NMNISTNet, self).__init__(tau_m, tau_s, n_steps)
        
        layer_configs = [
            (conv_pool_block, {'in_channels': 2, 'out_channels': 12, 'kernel_size': 5, 'padding': 1, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}),
            (conv_pool_block, {'in_channels': 12, 'out_channels': 64, 'kernel_size': 5, 'padding': 0, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}),
            (L.SNNLinear, {'in_features': 2304, 'out_features': 10, **self.tsslbp_config}),
        ]

        self.layers = nn.ModuleList(self._make_layers(layer_configs))
        
        if weight is not None:
            self.load_model(weight)

# DVSGesture Network 128x128 input
class DVSGestureNet(BaseNet):
    def __init__(self, tau_m, tau_s, n_steps, weight = None):
        super(DVSGestureNet, self).__init__(tau_m, tau_s, n_steps)
        
        layer_configs = [
            (conv_pool_block, {'in_channels': 2, 'out_channels': 15, 'kernel_size': 5, 'padding': 2, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}), 
            (conv_pool_block, {'in_channels': 15, 'out_channels': 40, 'kernel_size': 5, 'padding': 2, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}), 
            (conv_pool_block, {'in_channels': 40, 'out_channels': 80, 'kernel_size': 3, 'padding': 1, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}), 
            (conv_pool_block, {'in_channels': 80, 'out_channels': 160, 'kernel_size': 3, 'padding': 1, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}), 
            (conv_pool_block, {'in_channels': 160, 'out_channels': 320, 'kernel_size': 3, 'padding': 1, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}), 
            (L.SNNLinear, {'in_features': 5120, 'out_features': 512, **self.tsslbp_config}),
            (L.SNNLinear, {'in_features': 512, 'out_features': 11, **self.tsslbp_config}),
        ]

        self.layers = nn.ModuleList(self._make_layers(layer_configs))
        
        if weight is not None:
            self.load_model(weight)
            

# DVSGesture Network 64x64 input
class PGen4NetMini(BaseNet):
    def __init__(self, tau_m, tau_s, n_steps, weight = None):
        super(PGen4NetMini, self).__init__(tau_m, tau_s, n_steps)
        
        layer_configs = [
            (conv_pool_block, {'in_channels': 1, 'out_channels': 15, 'kernel_size': 5, 'padding': 2, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}), #32
            (conv_pool_block, {'in_channels': 15, 'out_channels': 40, 'kernel_size': 5, 'padding': 2, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}), #16
            (conv_pool_block, {'in_channels': 40, 'out_channels': 80, 'kernel_size': 3, 'padding': 1, 'pooling_size': 2, 'pooling_stride': 2, 'tsslbp_config': self.tsslbp_config}), #8
            (L.SNNConv3d, {'in_channels': 80, 'out_channels': 160, 'kernel_size': 3, 'padding': 1, 'stride': 2, **self.tsslbp_config}), # 4
            (conv_pool_block, {'in_channels': 160, 'out_channels': 320, 'kernel_size': 3, 'padding': 1, 'pooling_size': 4, 'pooling_stride': 4, 'tsslbp_config': self.tsslbp_config}), # ?
            (L.SNNLinear, {'in_features': 320, 'out_features': 64, **self.tsslbp_config}),
            (L.SNNLinear, {'in_features': 64, 'out_features': 5, **self.tsslbp_config}),
        ]

        self.layers = nn.ModuleList(self._make_layers(layer_configs))
        
        if weight is not None:
            self.load_model(weight)
