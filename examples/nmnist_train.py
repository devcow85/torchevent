import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader

from dvc import dataset as ds
from torchevent.transforms import RandomTemporalCrop, TemporalCrop, EventFrameMinMaxScaler, ToFrameAuto
from torchevent.utils import runner
from torchevent.dataset import HybridCachedDataset

if __name__ == "__main__":

    batch_size = 16

    train_ransform = transforms.Compose([
            RandomTemporalCrop(time_window = 33000),
            ToFrameAuto(n_time_bins=5, aspect_ratio=False),
            EventFrameMinMaxScaler(u8int=True)
            ])
            

    val_transform = transforms.Compose([
        TemporalCrop(time_window = 33000),
        ToFrameAuto(n_time_bins=5, aspect_ratio=False),
        EventFrameMinMaxScaler(u8int=True)
        ])

    train_set = ds.DVCDataset("NImagenet_vehicle", train=True, transforms=train_ransform)
    test_set = ds.DVCDataset("NImagenet_vehicle", train=False, transforms=val_transform)
    train_set_hcd = HybridCachedDataset(train_set, cache_size=500, num_workers=32, cache_path="hybrid_cache/NImagenet_vehicle/train")
    test_set_hcd = HybridCachedDataset(test_set, cache_size=500, num_workers=32, cache_path="hybrid_cache/NImagenet_vehicle/val")
    train_ds = DataLoader(train_set_hcd, shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=True)
    val_ds = DataLoader(test_set_hcd, shuffle=False, batch_size=100, num_workers=8)

    # train_recipes = {
    #     "model":{
    #         "model_name": "NMNISTNet",
    #         "kwargs": {"tau_m":5, "tau_s":1, "n_steps":5}
    #     },
    #     "optimizer":{
    #         "class": "AdamW",
    #         "kwargs": {"lr": 0.0005}
    #     },
    #     'loss': {
    #         "class": "SpikeCountLoss",
    #         "args": (4, 1)
    #     },
    #     'max_epoch': 100
    # }

    # runner(train_recipes, (train_ds, val_ds))
    
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
            "class": "SpikeCumulativeLoss",
            "args": ()
        },
        'max_epoch': 100
    }

    runner(train_recipes, (train_ds, val_ds))