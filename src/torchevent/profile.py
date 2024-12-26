import contextlib

import time
import pickle

from torchevent.utils import _tensor_to_numpy, _parse_extra_repr, _convert_state_dict_to_numpy


class LayerwiseProfiler:
    def __init__(self, model):
        self.pdata = []
        self.original_forwards = {}
        self.model = model
    
    def _wrapper(self, module, name, profile_type):
        """
        Wrap forward function to profile layer informations
        """
        
        original_forward = module.forward
        self.original_forwards[name] = (module, original_forward)  # Store the module and its original forward method
        
        def hook_fn(*args, **kwargs):
            start_time = time.time()
            output = original_forward(*args, **kwargs)
            elapsed_time = time.time() - start_time
            
            if profile_type == 'trace':
                layer_info = {
                    "module_type": type(module).__name__,
                    "inputs": tuple(_tensor_to_numpy(inp) for inp in args[0]),
                    "outputs": _tensor_to_numpy(output),
                    "extra_repr": _parse_extra_repr(module.extra_repr()),
                    "state_dict": _convert_state_dict_to_numpy(module.state_dict())
                }
            elif profile_type == 'summary':
                input_shape = args[0].size() if not isinstance(args[0], list) else args[0][0].size()
                output_shape = output.size()
                params = sum(p.numel() for p in module.parameters())

                layer_info = {
                    "module_type": type(module).__name__,
                    "module_name": name,
                    "input_shape": str(list(input_shape)),
                    "output_shape": str(list(output_shape)),
                    # "MAC": mac * 2,
                    "Params": params,
                    "Latency": elapsed_time*1e6,    # Latency in microseconds
                }
            
            # append profile data
            self.pdata.append(layer_info)
            return output
        
        module.forward = hook_fn
    
    @contextlib.contextmanager
    def profile(self, profile_type='summary'):
        """
        Profile all leaf modules in the registered model.
        Automatically clears profiling data after the context ends.
        """
        self.pdata.clear()
        
        if self.model is None:
            raise ValueError("Model is not set. Use `set_model()` to register a model.")

        for name, module in self.model.named_modules():
            if not list(module.children()):  # Only wrap leaf nodes
                self._wrapper(module, name, profile_type)

        try:
            yield self
        finally:
            self.clear()

    def clear(self):
        """
        Restore the original forward methods for all wrapped modules and clear profiler data.
        """
        for name, (module, original_forward) in self.original_forwards.items():
            module.forward = original_forward  # Restore the original forward method
        self.original_forwards.clear()
        
    def get_data(self):
        return self.pdata
    
    def to_pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.pdata, f)
    