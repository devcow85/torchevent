from typing import Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
import tonic
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as T

import torch.nn.functional as F
import torch
from PIL import Image


@dataclass
class RandomTemporalCrop:
    time_window: int = 99000
    padding_enable: bool = False

    def __call__(self, events):
        if events['t'][-1] - events['t'][0] < self.time_window:
            if self.padding_enable:
                dummy_event = np.array([(events['t'][0] + self.time_window, 0, 0, 0)], dtype=events.dtype)
                events = np.concatenate((events, dummy_event))
                events = np.sort(events, order='t')

            else:
                raise ValueError("Time window is too small")
        
        if events['t'][0] == events['t'][-1] - self.time_window:
            start_time = events['t'][0]
        else:
            start_time = np.random.randint(events['t'][0], events['t'][-1] - self.time_window)
        end_time = start_time + self.time_window

        return events[(events["t"] >= start_time) & (events["t"] <= end_time)]

@dataclass
class TemporalCrop:
    time_window: int = 99000
    padding_enable: bool = False
    
    def __call__(self, events):
        start_time = events['t'][0]
        end_time = start_time + self.time_window
        
        if events['t'][-1] < self.time_window:
            if self.padding_enable:
                dummy_event = np.array([(events['t'][0] + self.time_window, 0, 0, 0)], dtype=events.dtype)
                events = np.concatenate((events, dummy_event))
                events = np.sort(events, order='t')
            else:
                raise ValueError("Time window is too small")
            
        return events[(events["t"] >= start_time) & (events["t"] <= end_time)]

@dataclass(frozen=True)
class ToFrameAuto:
    time_window: Optional[float] = None
    event_count: Optional[int] = None
    n_time_bins: Optional[int] = None
    n_event_bins: Optional[int] = None
    overlap: float = 0
    include_incomplete: bool = False
    aspect_ratio: Optional[bool] = True

    def __call__(self, events):
        sensor_size = (max(events["x"]) + 1, max(events["y"]) + 1, 2)
        
        if self.aspect_ratio:
            x_max, y_max = max(events["x"]), max(events["y"])
            w_max = max(x_max, y_max) + 1
            sensor_size = (w_max, w_max, 2)
            
        return tonic.transforms.ToFrame(
            sensor_size=sensor_size,
            time_window=self.time_window,
            event_count=self.event_count,
            n_time_bins=self.n_time_bins,
            n_event_bins=self.n_event_bins,
            overlap=self.overlap,
            include_incomplete=self.include_incomplete,
        )(events)
        
@dataclass(frozen=True)
class MergeFramePolarity:
    bias: int = 128
    scale: float = 1.0

    def __call__(self, frames):
        merged_frames = np.zeros((frames.shape[0], 1,) + frames.shape[2:], dtype=np.int16)
    
        for i, frame in enumerate(frames):
            merged_frames[i][0] = self.bias + self.scale * (frame[1] - frame[0])
        
        return merged_frames


@dataclass(frozen=True)
class MinMaxScaler:
    min_val: float
    max_val: float
    
    def __call__(self, frame):
        return (frame - self.min_val) / (self.max_val - self.min_val)*255
        

@dataclass(frozen=True)
class EventFrameResize:
    size: tuple
    
    def __call__(self, frames):
        
        resized_frame = np.zeros(frames.shape[:2]+self.size[::-1], dtype=np.int16)
        for idx, frame in enumerate(frames):
            frame = frame.astype(np.uint8)
            pil_frame = to_pil_image(frame.transpose(1,2,0))
            
            resized_frame[idx] = pil_frame.resize(self.size)
        
        return resized_frame

@dataclass(frozen=True)
class EventFrameRandomResizedCrop:
    size: tuple  
    scale: tuple = (0.08, 1.0)  
    ratio: tuple = (3. / 4., 4. / 3.) 
    interpolation: int = Image.BILINEAR
    
    def __call__(self, frames):
        random_resized_crop = T.RandomResizedCrop(self.size, scale=self.scale, ratio=self.ratio, interpolation=self.interpolation)

        resized_frame = np.zeros(frames.shape[:2] + self.size[::-1], dtype=np.int16)

        for idx, frame in enumerate(frames):
            frame = frame.astype(np.uint8)
            pil_frame = to_pil_image(frame.transpose(1, 2, 0))

            pil_frame_cropped = random_resized_crop(pil_frame)

            resized_frame[idx] = pil_frame_cropped

        return resized_frame

@dataclass(frozen=True)
class EventFrameSumResize:
    size: tuple

    def __call__(self, frames):
        target_height, target_width = self.size

        resized_frame = np.zeros((frames.shape[0], frames.shape[1], target_height, target_width), dtype=np.int16)

        for idx, frame in enumerate(frames):
            frame_tensor = torch.from_numpy(frame).float()

            scale_y = frame.shape[1] // target_height
            scale_x = frame.shape[2] // target_width
            pooled_frame = F.avg_pool2d(frame_tensor, kernel_size=(scale_y, scale_x), stride=(scale_y, scale_x)) * (scale_y * scale_x)

            resized_frame[idx] = pooled_frame.numpy().astype(np.int16)

        return resized_frame
    
@dataclass(frozen=True)
class EventNormalize:
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    
    def __call__(self, frames):
        return (frames - self.mean) / self.std
  
@dataclass(frozen=True)
class EventFrameMinMaxScaler:
    scale: float = 1.0
    u8int: bool = True
    
    def __call__(self, frames):
        max_val = (np.max(frames, axis=(1,2,3), keepdims=True)+1e-8)
        if self.u8int:
            out = frames/max_val*255
            out = out.round()
            return out
        return frames/max_val*self.scale
    
    
@dataclass(frozen=True)
class UniformNoiseAuto:
    n: Union[int, Tuple[int,int]]
    
    def __call__(self, events):
        sensor_size = (max(events["x"]) + 1, max(events["y"]) + 1, 2)

        return tonic.transforms.UniformNoise(
            sensor_size=sensor_size,
            n = self.n
        )(events)