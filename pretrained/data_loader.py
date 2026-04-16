# -*- coding:utf-8 -*-
import bisect
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class TorchDataset(Dataset):
    """
    Memory-safe dataset for NPZ files.
    Chỉ build index theo subject/file, không nạp toàn bộ epoch vào RAM ngay từ đầu.
    Mỗi __getitem__ mới đọc đúng 1 epoch cần dùng từ file .npz.
    """

    def __init__(self, subj_list, npz_dir, scaler: bool = False):
        super().__init__()
        self.npz_dir = npz_dir
        self.scaler_flag = scaler
        self.file_infos = []
        self.cum_ends = []
        self.total_epochs = 0
        self._build_index(subj_list)

    def _build_index(self, subj_list):
        print(f"--- Build index cho {len(subj_list)} bệnh nhân từ {self.npz_dir} ---")

        for i, sid in enumerate(subj_list):
            if i % 10 == 0:
                print(f" > Index file thứ {i}/{len(subj_list)}: {sid}")

            path = os.path.join(self.npz_dir, sid if sid.endswith('.npz') else f"{sid}.npz")
            if not os.path.exists(path):
                print(f"   [Cảnh báo] Không tìm thấy: {path}")
                continue

            data = np.load(path, mmap_mode='r', allow_pickle=True)
            n_epochs = int(len(data['y']))
            if n_epochs <= 0:
                continue

            self.total_epochs += n_epochs
            self.file_infos.append({
                'path': path,
                'n_epochs': n_epochs,
            })
            self.cum_ends.append(self.total_epochs)

        print(f"--- Index xong! Tổng số epoch: {self.total_epochs} ---")

    def __len__(self):
        return self.total_epochs

    def _locate(self, global_idx: int):
        file_idx = bisect.bisect_right(self.cum_ends, global_idx)
        start = 0 if file_idx == 0 else self.cum_ends[file_idx - 1]
        local_idx = global_idx - start
        return self.file_infos[file_idx]['path'], local_idx

    def __getitem__(self, item):
        path, local_idx = self._locate(int(item))
        data = np.load(path, mmap_mode='r', allow_pickle=True)

        x = data['x'][local_idx].astype(np.float32, copy=False)   # (C, T)
        y = int(data['y'][local_idx])

        if self.scaler_flag:
            mean = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1, keepdims=True)
            x = (x - mean) / (std + 1e-6)

        return torch.from_numpy(x.copy()), torch.tensor(y, dtype=torch.long)
