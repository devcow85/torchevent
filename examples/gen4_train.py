import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader

from dvc import dataset as ds
from torchevent.transforms import RandomTemporalCrop, TemporalCrop, EventFrameMinMaxScaler, ToFrameAuto, EventFrameRandomResizedCrop, MergeFramePolarity, UniformNoiseAuto, EventNormalize
from torchevent.utils import runner
from torchevent.dataset import HybridCachedDataset
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":

    batch_size = 16

    train_ransform = transforms.Compose([
            RandomTemporalCrop(time_window = 66000, padding_enable=True),
            UniformNoiseAuto(n=(100, 300)),
            ToFrameAuto(n_time_bins=5, aspect_ratio=False),
            MergeFramePolarity(),
            EventFrameRandomResizedCrop((64,64)),
            EventNormalize(mean=(128,), std=(1,))
            ])
            

    val_transform = transforms.Compose([
            TemporalCrop(time_window = 66000, padding_enable=True),
            ToFrameAuto(n_time_bins=5, aspect_ratio=False),
            MergeFramePolarity(),
            EventFrameRandomResizedCrop((64,64)),
            EventNormalize(mean=(128,), std=(1,))
            ])

    train_set = ds.DVCDataset("gen4ad_mini", train=True, transforms=train_ransform)
    test_set = ds.DVCDataset("gen4ad_mini", train=False, transforms=val_transform)
    # train_set_hcd = HybridCachedDataset(train_set, cache_size=500, num_workers=32, cache_path="hybrid_cache/Gen4ADm/train")
    # test_set_hcd = HybridCachedDataset(test_set, cache_size=500, num_workers=32, cache_path="hybrid_cache/Gen4ADm/val")
    train_ds = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=True)
    val_ds = DataLoader(test_set, shuffle=False, batch_size=100, num_workers=8)

    train_recipes = {
        "model":{
            "model_name": "PGen4NetMini",
            "kwargs": {"tau_m":5, "tau_s":1, "n_steps":5}
        },
        "optimizer":{
            "class": "AdamW",
            "kwargs": {"lr": 0.001}
        },
        'loss': {
            "class": "SpikeCountLoss",
            "args": (4, 1)
        },
        # 'loss': {
        #     "class": "SpikeCumulativeLoss",
        #     "args": ()
        # },
        'max_epoch': 200
    }

    artifact = runner(train_recipes, (train_ds, val_ds))
    artifact.export("results/gen4adm_countloss") 