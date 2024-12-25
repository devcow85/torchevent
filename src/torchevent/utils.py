import random
import re

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
# from tqdm import tqdm

import matplotlib.pyplot as plt

def set_seed(random_seed):
    """reproducible option

    Args:
        random_seed (int): seed value
    """
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
    """
    Convert spike train data into summed data or predicted classes.

    Args:
        spikes (torch.Tensor): Input tensor of shape (Batch, num_class, 1, 1, n_step).
        return_pred (bool): If True, returns the predicted class index for each batch.

    Returns:
        torch.Tensor: If `return_pred` is False, returns a tensor of shape (Batch, num_class).
                      If `return_pred` is True, returns a tensor of shape (Batch,).
    """
    data = torch.sum(spikes, dim=4).squeeze_(-1).squeeze_(-1)
    
    if return_pred:
        return data.argmax(axis=1)
    
    return data

def _parse_extra_repr(extra_repr_str):
    # Split the string by commas while keeping the text inside parentheses together
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
                kwargs[key] = value  # For strings and unrecognized types
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
    """데이터를 0-255 사이로 정규화하고, uint8로 변환"""
    data_min = np.min(data)
    data_max = np.max(data)
    # 데이터가 이미 동일한 값일 경우 0으로 나눌 수 없으니, 이 경우 대비
    if data_max != data_min:
        normalized_data = (data - data_min) / (data_max - data_min) * 255.0
    else:
        normalized_data = np.zeros_like(data)
    return normalized_data.astype(np.uint8)

def plot_event_frame(event_data, file_name):
    n_step, ch, width, height = event_data.shape
    if ch == 1:
        # merge pol
        frame_concat = np.concatenate([event_data[j,0] for j in range(n_step)], axis=1)
    else:
        frame_concat = np.concatenate([
            np.stack([event_data[j, 0], event_data[j, 1], np.zeros((width, height))], axis=-1)  # R=polarity 0, G=polarity 1, B=0
            for j in range(n_step)
        ], axis=1)
    
    
    # 수평으로 이어 붙인 이미지들을 다시 수직으로 이어 붙임
    i8_data = to_uint8(frame_concat)
    
    plt.imsave(file_name, i8_data)
    
def train(model, epochs, data_loader, optimizer, loss, device = "cuda"):
    model.train()
    model.to(device)
    
    total_acc = 0
    total_loss = 0
    total_len = 0
    
    with tqdm(data_loader, unit="batch", ) as nbatch:
        nbatch.set_description(f'Epoch - {epochs}')
        for data, targets in nbatch:
            data = data.to(torch.float32)
            data, targets = data.to(device), targets.to(device)
                        
            optimizer.zero_grad()
            outputs = model(data)
            
            output_spike = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
            
            vloss = loss(outputs, targets)
            vloss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            # model.weight_clipper()
            
            total_loss+= torch.sum(vloss).item()
            total_len+=len(targets)
            
            total_acc+=(
                (output_spike.argmax(axis=1) == targets.cpu().numpy())
                .sum()
                .item()
            )
            
            nbatch.set_postfix_str(f"train acc: {total_acc / total_len:.3f}, train loss: {total_loss / total_len:.3f}")

        return total_acc, total_loss, total_len
    
    
def validation(model, data_loader, device):
    total_acc = 0
    total_len = 0
    all_targets = []
    all_preds = []
    
    model.eval()
    
    with tqdm(data_loader, unit="batch") as nbatch:
        for data, targets in nbatch:
            data = data.to(torch.float32)
            data, targets = data.to(device), targets.to(device)
            
            with torch.no_grad():
                outputs = model(data)
            
            output_spike = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
            total_len+=len(targets)
            
            preds = output_spike.argmax(axis=1)
            
            total_acc+=(
                (preds == targets.cpu().numpy())
                .sum()
                .item()
            )
            
            all_targets.extend(targets.detach().cpu().numpy())  # Ground truth values
            all_preds.extend(preds)  # Predicted values

            nbatch.set_postfix_str(f"val acc: {total_acc / total_len:.3f}")
            
    return total_acc, total_len, (all_targets, all_preds)