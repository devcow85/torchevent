import random
import re

import numpy as np
import torch
from tqdm import tqdm
from torchevent.dataset import HybridCachedDataset
import matplotlib.pyplot as plt

def set_seed(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def expand_to_3d(value, name, last_dim = 1):
    if isinstance(value, int):
        return (value, value, last_dim)
    elif len(value) == 2:
        return (value[0], value[1], last_dim)
    else:
        raise ValueError(f"{name} must be int or tuple of size 2, got {value}")

def weight_clipper(weight, clip_value=4):
    with torch.no_grad():
        weight.clamp_(-clip_value, clip_value)

def spike2data(spikes, return_pred = False):
    data = torch.sum(spikes, dim=4).squeeze_(-1).squeeze_(-1)
    
    if return_pred:
        return data.argmax(axis=1)
    
    return data

def _parse_extra_repr(extra_repr_str):
    parts = re.split(r',\s*(?![^()]*\))', extra_repr_str)
    
    args = []
    kwargs = {}
    
    for part in parts:
        if '=' in part:
            key, value = part.split('=', 1)
            key = key.strip()
            value = value.strip()
            try:
                kwargs[key] = eval(value)
            except NameError:
                kwargs[key] = value
        else:
            try:
                args.append(eval(part.strip()))
            except NameError:
                args.append(part.strip())
    
    return args, kwargs

def _tensor_to_numpy(tensor):
    if tensor.is_quantized:
        tensor = tensor.int_repr().float()
    return tensor.detach().cpu().numpy()

def _convert_state_dict_to_numpy(state_dict):
    numpy_state_dict = {}
    for key, value in state_dict.items():
        numpy_state_dict[key] = _tensor_to_numpy(value) if isinstance(value, torch.Tensor) else value
    return numpy_state_dict

def to_uint8(data):
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max != data_min:
        normalized_data = (data - data_min) / (data_max - data_min) * 255.0
    else:
        normalized_data = np.zeros_like(data)
    return normalized_data.astype(np.uint8)

def plot_event_frame(event_data, file_name):
    n_step, ch, width, height = event_data.shape
    if ch == 1:
        frame_concat = np.concatenate([event_data[j,0] for j in range(n_step)], axis=1)
    else:
        frame_concat = np.concatenate([
            np.stack([event_data[j, 0], event_data[j, 1], np.zeros((width, height))], axis=-1)
            for j in range(n_step)
        ], axis=1)
    
    i8_data = to_uint8(frame_concat)
    
    plt.imsave(file_name, i8_data)

def mlloops(model, data_loader, optimizer = None, criterion = None, device = 'cpu', phase = 'train', metric_hook = None):
    if phase not in ['train', 'eval']:
        raise ValueError(f"Wrong {phase} is entered, please check phase value again between 'train', 'eval'")
    
    getattr(model, phase)()
    model.to(device)
    
    total_loss = 0
    total_samples = 0
    
    all_outputs = []
    all_targets = []
    
    metric_dict = {}  
    
    with tqdm(data_loader, unit="batch", desc=phase) as nbatch:
        for data, targets in nbatch:
            data, targets = data.to(device), targets.to(device)
            
            data = data.to(torch.float32)
            
            if phase == 'train':
                optimizer.zero_grad()
                
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            if phase == 'train':
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                model.weight_clipper()
            
            total_samples += data.size(0)
            
            all_outputs.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())

            metric_dict.update(metric_hook(all_outputs, all_targets) if metric_hook else {})
            metric_dict["loss"] = metric_dict.get("loss", 0) + loss.item()
            
            nbatch.set_postfix(metric_dict)
    
    metric_dict["loss"] /= total_samples
    metric_dict['elapsed_time'] = nbatch.format_dict['elapsed']
            
    return metric_dict

from torchevent import models, loss
from torchevent.artifacts import ArtifactManager
from torchevent.metrics import extented_cls_metric_hook, acc_metric_hook
import copy

def runner(train_recipes, dataloader):
    train_loader, val_loader = dataloader
    
    model = getattr(models, train_recipes["model"]["model_name"])(**train_recipes["model"]["kwargs"])
    
    artifacts = ArtifactManager(model._get_name())
    artifacts['summary'] = model.summary(train_loader.dataset[0][0].shape)   # 01. dataframe for given dataset's sensor size
    
    artifacts['train_recipes'] = train_recipes  # 07. training recipes
    
    optimizer = getattr(torch.optim, train_recipes["optimizer"]["class"])(model.parameters(), **train_recipes["optimizer"]["kwargs"])
    criterion = getattr(loss,train_recipes["loss"]["class"])(*train_recipes["loss"]["args"])
    # criterion = loss.SpikeCumulativeLoss(gamma=0)
    
    learning_log = []
    best_acc = 0
    
    patience          = 3        # 개선 없이 기다릴 Epoch 수
    epochs_no_improve = 0        # 연속 미개선 Epoch 카운터
    generation        = 0        # 캐시 세대 기록(선택)

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
            epochs_no_improve = 0
            # model.save_model('examples/best_model.pt')  # 03. best model
            artifacts['best_model'] = copy.deepcopy(model.state_dict())
            print("save model!")
        else:
            epochs_no_improve += 1

        if (epochs_no_improve >= patience and
            isinstance(train_loader.dataset, HybridCachedDataset)):
            artifacts['learning_curves'] = learning_log    # 02. learning logs

            generation += 1
            train_loader.dataset.refresh_cache()
            
            print(f"\n🔄 Accuracy stagnant for {patience} epochs → "
              f"refreshing cached dataset (gen {generation}) ...")
            
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
    
    return artifacts