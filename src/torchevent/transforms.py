from typing import Optional
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
        )(events) # for channel first format
        
@dataclass(frozen=True)
class MergeFramePolarity:
    bias: int = 128
    scale: float = 1.0  # Add a scaling factor with default value 1.0

    def __call__(self, frames):
        merged_frames = np.zeros((frames.shape[0], 1,) + frames.shape[2:], dtype=np.int16)
    
        for i, frame in enumerate(frames):
            # Apply the scale to the difference between frame[1] and frame[0]
            merged_frames[i][0] = self.bias + self.scale * (frame[1] - frame[0])
        
        return merged_frames  # channel first format


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
        
        return resized_frame  # Stack frames back into a single tensor

@dataclass(frozen=True)
class EventFrameRandomResizedCrop:
    size: tuple  # 목표 크기 (width, height)
    scale: tuple = (0.08, 1.0)  # 크롭할 이미지 크기의 범위 비율 (default는 RandomResizedCrop의 기본값)
    ratio: tuple = (3. / 4., 4. / 3.)  # 크롭할 이미지의 가로 세로 비율 범위
    interpolation: int = Image.BILINEAR  # 리사이즈 시 사용할 보간법
    
    def __call__(self, frames):
        """
        frames: (N, H, W, C) 형식의 여러 개의 프레임을 입력받아 RandomResizedCrop으로 크롭하고, 다시 target size로 resize.
        """
        # RandomResizedCrop 설정: 입력된 파라미터들을 활용
        random_resized_crop = T.RandomResizedCrop(self.size, scale=self.scale, ratio=self.ratio, interpolation=self.interpolation)

        resized_frame = np.zeros(frames.shape[:2] + self.size[::-1], dtype=np.int16)

        for idx, frame in enumerate(frames):
            frame = frame.astype(np.uint8)
            pil_frame = to_pil_image(frame.transpose(1, 2, 0))  # frame을 PIL 이미지로 변환

            # RandomResizedCrop을 적용하여 크롭 및 리사이즈
            pil_frame_cropped = random_resized_crop(pil_frame)

            # 결과를 resized_frame 배열에 저장
            resized_frame[idx] = pil_frame_cropped

        return resized_frame  # Stack frames back into a single tensor

@dataclass(frozen=True)
class EventFrameSumResize:
    size: tuple  # target size (width, height)

    def __call__(self, frames):
        # frames are assumed to be in shape (batch_size, channels, height, width)

        # Get the target size
        target_height, target_width = self.size

        # Prepare an empty array to hold the resized frames
        resized_frame = np.zeros((frames.shape[0], frames.shape[1], target_height, target_width), dtype=np.int16)

        for idx, frame in enumerate(frames):
            # Apply sum pooling by reducing the resolution using summation over pooling windows
            frame_tensor = torch.from_numpy(frame).float()  # Convert the frame to a torch tensor

            # Perform sum pooling using a kernel size corresponding to the downscaling factor
            scale_y = frame.shape[1] // target_height
            scale_x = frame.shape[2] // target_width
            pooled_frame = F.avg_pool2d(frame_tensor, kernel_size=(scale_y, scale_x), stride=(scale_y, scale_x)) * (scale_y * scale_x)

            # Convert back to numpy and store it in resized_frame
            resized_frame[idx] = pooled_frame.numpy().astype(np.int16)

        return resized_frame  # Return the resized frames
    
@dataclass(frozen=True)
class EventNormalize:
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    
    def __call__(self, frames):
        mean = torch.tensor(self.mean).view(1, -1, 1, 1)
        std = torch.tensor(self.std).view(1, -1, 1, 1)
        return (frames - mean) / std