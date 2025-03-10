import pytest
import numpy as np
import torch
from torchevent.transforms import (
    RandomTemporalCrop, TemporalCrop, ToFrameAuto, MergeFramePolarity,
    MinMaxScaler, EventFrameResize, EventFrameRandomResizedCrop, EventFrameSumResize, EventNormalize
)

# def test_random_temporal_crop():
#     events = np.array([(0, 1, 1, 1), (99000, 1, 1, 1)], dtype=[('t', np.int64), ('x', np.int64), ('y', np.int64), ('p', np.int64)])
#     crop = RandomTemporalCrop(time_window=50000)
#     cropped_events = crop(events)
#     assert len(cropped_events) > 0, "RandomTemporalCrop should not return empty events"

def test_temporal_crop():
    events = np.array([(0, 1, 1, 1), (99000, 1, 1, 1)], dtype=[('t', np.int64), ('x', np.int64), ('y', np.int64), ('p', np.int64)])
    crop = TemporalCrop(time_window=50000)
    cropped_events = crop(events)
    assert len(cropped_events) > 0, "TemporalCrop should not return empty events"

def test_to_frame_auto():
    events = np.array([(0, 1, 1, 1), (10, 2, 2, -1)], dtype=[('t', np.int64), ('x', np.int64), ('y', np.int64), ('p', np.int64)])
    frame = ToFrameAuto(time_window=100)(events)
    assert frame is not None, "ToFrameAuto should return a valid frame"

def test_merge_frame_polarity():
    frames = np.random.randint(0, 255, (5, 2, 32, 32), dtype=np.int16)
    merge = MergeFramePolarity()
    merged_frame = merge(frames)
    assert merged_frame.shape == (5, 1, 32, 32), "Merged frame shape is incorrect"

# def test_min_max_scaler():
#     scaler = MinMaxScaler(min_val=0, max_val=255)
#     frame = np.random.randint(0, 255, (32, 32), dtype=np.int16)
#     scaled_frame = scaler(frame)
#     assert np.max(scaled_frame) == 255, "Max value of scaled frame should be 255"
#     assert np.min(scaled_frame) == 0, "Min value of scaled frame should be 0"

def test_event_frame_resize():
    frames = np.random.randint(0, 255, (5, 1, 32, 32), dtype=np.int16)
    resize = EventFrameResize(size=(16, 16))
    resized_frames = resize(frames)
    assert resized_frames.shape == (5, 1, 16, 16), "Resized frame shape is incorrect"

def test_event_frame_random_resized_crop():
    frames = np.random.randint(0, 255, (5, 1, 32, 32), dtype=np.int16)
    crop = EventFrameRandomResizedCrop(size=(16, 16))
    cropped_frames = crop(frames)
    assert cropped_frames.shape == (5, 1, 16, 16), "Cropped frame shape is incorrect"

def test_event_frame_sum_resize():
    frames = np.random.randint(0, 255, (5, 1, 32, 32), dtype=np.int16)
    sum_resize = EventFrameSumResize(size=(16, 16))
    resized_frames = sum_resize(frames)
    assert resized_frames.shape == (5, 1, 16, 16), "Sum-resized frame shape is incorrect"

def test_event_normalize():
    frames = torch.rand(5, 1, 32, 32).numpy()
    normalize = EventNormalize(mean=(128,), std=(1,))
    normalized_frames = normalize(frames)
    assert normalized_frames.shape == frames.shape, "Normalized frame shape is incorrect"