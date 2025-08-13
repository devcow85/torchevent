import tonic
import tonic.transforms as transforms
import torch
from torch.utils.data import DataLoader

from torchevent.utils import set_seed, mlloops
from torchevent import models, loss
from torchevent.transforms import RandomTemporalCrop, TemporalCrop
from torchevent.metrics import extented_cls_metric_hook, acc_metric_hook
from torchevent.artifacts import ArtifactManager
import copy

if __name__ == "__main__":
    set_seed(7)
    
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
    
    model = getattr(models, train_recipes["model"]["model_name"])(**train_recipes["model"]["kwargs"])
    
    artifacts = ArtifactManager(model._get_name())
    artifacts['summary'] = model.summary(train_ds[0][0].shape)   # 01. dataframe for given dataset's sensor size
    
    artifacts['train_recipes'] = train_recipes  # 07. training recipes
    
    optimizer = getattr(torch.optim, train_recipes["optimizer"]["class"])(model.parameters(), **train_recipes["optimizer"]["kwargs"])
    criterion = getattr(loss,train_recipes["loss"]["class"])(*train_recipes["loss"]["args"])
    # criterion = loss.SpikeCumulativeLoss(gamma=0)
    
    learning_log = []
    best_acc = 0
    for epoch in range(train_recipes["max_epoch"]):
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
            # model.save_model('examples/best_model.pt')  # 03. best model
            artifacts['best_model'] = copy.deepcopy(model.state_dict())
            print("save model!")
    
    artifacts['learning_curves'] = learning_log    # 02. learning logs
    
    model.load_state_dict(artifacts['best_model'])
    print('best eval result')
    metric = mlloops(model, val_loader, optimizer, criterion, "cuda", 'eval', extented_cls_metric_hook)
    
    artifacts['eval_metric'] = metric   # 04. evaluation metric
    
    print('trace model')
    for data, target in val_loader:
        data = data.to('cuda').to(torch.float32)
        trace_result = model.trace(data)
        
        break
    
    artifacts['trace_data'] = trace_result  # 06. trace data
    artifacts.save()