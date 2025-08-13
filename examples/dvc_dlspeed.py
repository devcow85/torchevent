import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader

from dvc import dataset as ds
from torchevent.transforms import RandomTemporalCrop, TemporalCrop, EventFrameMinMaxScaler
from torchevent.utils import runner
from torchevent.dataset import HybridCachedDataset
from tqdm import tqdm

if __name__ == "__main__":

    batch_size = 64

    train_ransform = transforms.Compose([
            RandomTemporalCrop(time_window = 99000),
            transforms.ToFrame(sensor_size=(tonic.datasets.NMNIST.sensor_size),
                            n_time_bins=5),
            EventFrameMinMaxScaler(scale=4)
            ])
            

    val_transform = transforms.Compose([
        TemporalCrop(time_window = 99000),
        transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size, 
                        n_time_bins=5),
        EventFrameMinMaxScaler(scale=4)
        ])

    train_set = ds.DVCDataset("NMNIST", train=True, transforms=train_ransform)
    test_set = ds.DVCDataset("NMNIST", train=False, transforms=val_transform)
    train_set_hcd = HybridCachedDataset(train_set, cache_size=500, num_workers=8)
    train_ds = DataLoader(train_set_hcd, shuffle=True, batch_size=batch_size, num_workers=16, pin_memory=True)
    val_ds = DataLoader(test_set, shuffle=False, batch_size=100, num_workers=8)

    with tqdm(train_ds, unit="batch") as nbatch:
        for data, targets in nbatch:
            pass