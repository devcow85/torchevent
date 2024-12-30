import time
import tonic
import tonic.transforms as transforms
import torch
from torch.utils.data import DataLoader

from torchevent.utils import set_seed, mlloops, spike2data
from torchevent import models, loss
from torchevent.transforms import RandomTemporalCrop, TemporalCrop
from torchevent.metrics import extented_cls_metric_hook, acc_metric_hook

import pandas as pd

if __name__ == "__main__":
    set_seed(7)
    
    # load dataset
    transform = transforms.Compose([
        RandomTemporalCrop(time_window = 99000),
        transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size,
                           n_time_bins=5)
    ])
    
    train_ds = tonic.datasets.NMNIST(save_to = 'data', 
                                         train=True, 
                                         transform = transform)
    
    transform = transforms.Compose([
        TemporalCrop(time_window = 99000),
        transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size,
                           n_time_bins=5)
    ])
    
    val_ds = tonic.datasets.NMNIST(save_to = 'data', 
                                         train=False, 
                                         transform = transform)
    
    batch_size = 32
    
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size, num_workers=8)
    
    model = models.NMNISTNet(5, 1, n_steps = 5)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    # criterion = loss.SpikeCountLoss(4, 1)
    criterion = loss.SpikeCumulativeLoss(gamma=0)
    
    
    artifacts = {}
    # artifacts dict
    # 1. model summary .. done
    # 2. learning curves .. done
    # 3. best model .. done
    # 4. evaluation result summary (f1, confusion map)  .. done
    # 5. inference layerwise profiling info (cpu, gpu)
    # 6. trace data .. done
    # 7. training recipes .. done
    summary_df = model.summary((5,2,34,34))
    
    artifacts['summary'] = summary_df   # 01. dataframe for given dataset's sensor size
    
    train_recipes = {
        "optimizer": 'AdamW(lr=0.0005)',
        'loss': 'SpikeCountLoss(desired=4, undesired=1)',
        'max_epoch': 1
    }
    artifacts['train_recipes'] = train_recipes  # 07. training recipes
    
    num_epochs = 1
    learning_log = []
    best_acc = 0
    for epoch in range(num_epochs):
        logdict = {'epoch': epoch, 'phase': 'train'}
        metric = mlloops(model, train_loader, optimizer, criterion, "cuda", 'train', acc_metric_hook)
        logdict.update(metric)
        learning_log.append(logdict)
        
        logdict = {'epoch': epoch, 'phase': 'eval'}
        metric = mlloops(model, val_loader, optimizer, criterion, "cuda", 'eval', extented_cls_metric_hook)
        logdict.update(metric)
        learning_log.append(logdict)
        
        if best_acc < float(metric['acc']):
            best_acc = float(metric['acc'])
            model.save_model('examples/best_model.pt')  # 03. best model
    
    artifacts['learning_curves'] = learning_log    # 02. learning logs
    
    model.load_model('examples/best_model.pt')  # reload best model weight
    print('best eval result')
    metric = mlloops(model, val_loader, optimizer, criterion, "cuda", 'eval', extented_cls_metric_hook)
    
    
    artifacts['eval_metric'] = metric   # 04. evaluation metric
    
    print('trace model')
    for data, target in val_loader:
        data = data.to('cuda').to(torch.float32)
        trace_result = model.trace(data)
        
        break
    
    artifacts['trace_data'] = trace_result  # 06. trace data