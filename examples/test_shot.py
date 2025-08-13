import tonic
import tonic.transforms as transforms
import torch
from torch.utils.data import DataLoader


from torchevent.transforms import RandomTemporalCrop, TemporalCrop
from torchevent.utils import runner
import copy

if __name__ == "__main__":
    
    # === load dataset ======================
    transform = transforms.Compose([
        RandomTemporalCrop(time_window = 99000),
        transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size,
                           n_time_bins=5)
    ])
    
    path = '/data/vision_data'
    train_ds = tonic.datasets.NMNIST(save_to = path, 
                                         train=True, 
                                         transform = transform)
    
    transform = transforms.Compose([
        TemporalCrop(time_window = 99000),
        transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size,
                           n_time_bins=5)
    ])
    
    val_ds = tonic.datasets.NMNIST(save_to = path, 
                                         train=False, 
                                         transform = transform)
    
    batch_size = 32
    
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size, num_workers=8)
    
    
    # === load model and train setup =========
    
    train_recipes = {
        "model":{
          "model_name": "NMNISTNet",
          "kwargs": {"tau_m":5, "tau_s":1, "n_steps":5}
        },
        "optimizer":{
            "class": "AdamW",
            "kwargs": {"lr": 0.0005}
        },
        'loss': {
            "class": "SpikeCountLoss",
            "args": (4, 1)
        },
        'max_epoch': 10
    }
    
    runner(train_recipes, (train_loader, val_loader))