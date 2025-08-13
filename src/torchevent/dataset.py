import os
import torch
import threading
import queue
from torch.utils.data import Dataset
from collections import OrderedDict
import os
import torch
import multiprocessing
import threading
import queue
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict


class HybridCachedDataset(Dataset):
    def __init__(self, original_dataset, cache_size=1000, cache_path="hybrid_cache", num_workers=4, preload=True):
        self.dataset = original_dataset
        self.cache_size = cache_size  # 메모리에 저장할 최대 샘플 개수
        self.cache = OrderedDict()  # LRU 캐시
        self.cache_path = cache_path
        self.num_workers = num_workers  # 병렬 처리할 프로세스 개수

        os.makedirs(self.cache_path, exist_ok=True)

        cached_files = set(f for f in os.listdir(self.cache_path) if f.endswith(".pt"))
        if len(cached_files) >= len(self.dataset):
            print(f"🚀 Found {len(cached_files)} cached samples. Skipping preloading.")
        else:
            print(f"⚡ Found {len(cached_files)} cached samples. Preloading missing data...")
            self._preload_to_disk_parallel()

        self.load_queue = queue.Queue()
        self.loader_thread = threading.Thread(target=self._async_loader, daemon=True)
        self.loader_thread.start()

    def _save_sample(self, idx):
        """하나의 샘플을 변환 후 저장 (병렬 처리 대상)"""
        torch.save(self.dataset[idx], os.path.join(self.cache_path, f"{idx}.pt"))

    def _preload_to_disk_parallel(self):
        """데이터를 병렬적으로 디스크에 저장"""
        print(f"Parallel preloading dataset using {self.num_workers} workers...")
        with multiprocessing.Pool(self.num_workers) as pool:
            pool.map(self._save_sample, range(len(self.dataset)))

    def _async_loader(self):
        """비동기적으로 데이터를 메모리에 로드"""
        while True:
            idx = self.load_queue.get()
            if idx is None:
                break
            if idx not in self.cache:
                data = torch.load(os.path.join(self.cache_path, f"{idx}.pt"))
                self._add_to_cache(idx, data)

    def _add_to_cache(self, idx, data):
        """메모리 캐시 관리 (LRU 방식)"""
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))  # 가장 오래된 데이터 제거
        self.cache[idx] = data

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        # 없으면 디스크에서 로드
        data = torch.load(os.path.join(self.cache_path, f"{idx}.pt"))
        self._add_to_cache(idx, data)

        # 다음 데이터 미리 로드 요청
        next_idx = (idx + 1) % len(self.dataset)
        if next_idx not in self.cache:
            self.load_queue.put(next_idx)

        return data

    def __len__(self):
        return len(self.dataset)

    def refresh_cache(self):
        print("🔄 Refreshing cache...")
        self.load_queue.put(None)  # 현재 로드 중인 작업을 중지
        self.loader_thread.join()  # 로드 스레드가 종료될 때까지 대기
        
        self.cache.clear()  # 캐시 비우기
        
        for f in os.listdir(self.cache_path):
            if f.endswith(".pt"):
                os.remove(os.path.join(self.cache_path, f)) 
                
        self._preload_to_disk_parallel()

        # restart the loader thread
        self.load_queue = queue.Queue()
        self.loader_thread = threading.Thread(
            target=self._async_loader, daemon=True)
        self.loader_thread.start()