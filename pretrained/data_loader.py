# # -*- coding:utf-8 -*-
# import mne
# import torch
# import random
# import numpy as np
# from torch.utils.data import Dataset
# import warnings

# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# random_seed = 777
# np.random.seed(random_seed)
# torch.manual_seed(random_seed)
# random.seed(random_seed)


# class TorchDataset(Dataset):
#     def __init__(self, paths, sfreq, rfreq, scaler: bool = False):
#         super().__init__()
#         self.x, self.y = self.get_data(paths, sfreq, rfreq, scaler)
#         self.x, self.y = torch.tensor(self.x, dtype=torch.float32), torch.tensor(self.y, dtype=torch.long)

#     @staticmethod
#     def get_data(paths, sfreq, rfreq, scaler_flag):
#         info = mne.create_info(sfreq=sfreq, ch_types='eeg', ch_names=['Fp1'])
#         scaler = mne.decoding.Scaler(info=info, scalings='median')
#         total_x, total_y = [], []
#         for path in paths:
#             data = np.load(path)
#             x, y = data['x'], data['y']
#             x = np.expand_dims(x, axis=1)
#             if scaler_flag:
#                 x = scaler.fit_transform(x)
#             x = mne.EpochsArray(x, info=info)
#             x = x.resample(rfreq)
#             x = x.get_data().squeeze()
#             total_x.append(x)
#             total_y.append(y)
#         total_x, total_y = np.concatenate(total_x), np.concatenate(total_y)
#         return total_x, total_y

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, item):
#         x = torch.tensor(self.x[item])
#         y = torch.tensor(self.y[item])
#         return x, y

# -*- coding:utf-8 -*-
import torch
import numpy as np
from torch.utils.data import Dataset
import os

class TorchDataset(Dataset):
    def __init__(self, subj_list, npz_dir, scaler: bool = False):
        super().__init__()
        self.npz_dir = npz_dir
        self.scaler_flag = scaler
        self.x, self.y = self.load_all_data(subj_list)
        
        # Chuyển thành tensor (N, Channels, T)
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def load_all_data(self, subj_list):
        total_x, total_y = [], []
        print(f"--- Bắt đầu nạp {len(subj_list)} bệnh nhân từ {self.npz_dir} ---")
        
        for i, sid in enumerate(subj_list):
            # 1. In tiến trình thực tế (Nạp đến đâu in đến đó)
            if i % 10 == 0: 
                print(f" > Đang xử lý file thứ {i}/{len(subj_list)}: {sid}")
            
            # 2. Xử lý đường dẫn
            if sid.endswith('.npz'):
                path = os.path.join(self.npz_dir, sid)
            else:
                path = os.path.join(self.npz_dir, f"{sid}.npz")
                
            if not os.path.exists(path):
                print(f"   [Cảnh báo] Không tìm thấy: {path}")
                continue
            
            # 3. Nạp dữ liệu thực tế (Bước này tốn thời gian nhất)
            data = np.load(path, allow_pickle=True)
            x, y = data['x'], data['y']
            
            # Chuẩn hóa (Z-score) nếu scaler_flag = True
            if self.scaler_flag:
                mean = x.mean(axis=-1, keepdims=True)
                std = x.std(axis=-1, keepdims=True)
                x = (x - mean) / (std + 1e-6)
                
            total_x.append(x)
            total_y.append(y)
            
        print(f"--- Nạp xong! Đang gộp dữ liệu... ---")
        return np.concatenate(total_x), np.concatenate(total_y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.x[item], self.y[item]