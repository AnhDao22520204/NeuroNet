# # -*- coding:utf-8 -*-
# import os
# import mne
# import torch
# import random
# import argparse
# import warnings
# import numpy as np
# import torch.nn as nn
# from typing import List
# import torch.optim as opt
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import accuracy_score, f1_score
# from models.neuronet.model import NeuroNet, NeuroNetEncoderWrapper


# warnings.filterwarnings(action='ignore')


# random_seed = 777
# np.random.seed(random_seed)
# torch.manual_seed(random_seed)
# random.seed(random_seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# def get_args():
#     file_name = 'mini'
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--n_fold', default=0, choices=[0, 1, 2, 3, 4])
#     parser.add_argument('--ckpt_path', default=os.path.join('..', '..', '..', 'ckpt',
#                                                             'ISRUC-Sleep', 'cm_eeg', file_name), type=str)
#     parser.add_argument('--epochs', default=300, type=int)
#     parser.add_argument('--batch_size', default=512, type=int)
#     parser.add_argument('--lr', default=0.00005, type=float)
#     return parser.parse_args()


# class Classifier(nn.Module):
#     def __init__(self, backbone, backbone_final_length):
#         super().__init__()
#         self.backbone = self.freeze_backbone(backbone)
#         self.backbone_final_length = backbone_final_length
#         self.feature_num = self.backbone_final_length * 2
#         self.dropout_p = 0.5
#         self.fc = nn.Sequential(
#             nn.Linear(backbone_final_length, self.feature_num),
#             nn.BatchNorm1d(self.feature_num),
#             nn.ELU(),
#             nn.Dropout(self.dropout_p),
#             nn.Linear(self.feature_num, 5)
#         )

#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.fc(x)
#         return x

#     @staticmethod
#     def freeze_backbone(backbone: nn.Module):
#         for name, module in backbone.named_modules():
#             for param in module.parameters():
#                 param.requires_grad = False
#         return backbone


# class Trainer(object):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.ckpt_path = os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'model', 'best_model.pth')
#         self.ckpt = torch.load(self.ckpt_path, map_location='cpu')
#         self.sfreq, self.rfreq = self.ckpt['hyperparameter']['sfreq'], self.ckpt['hyperparameter']['rfreq']
#         self.ft_paths, self.eval_paths = self.ckpt['paths']['ft_paths'], self.ckpt['paths']['eval_paths']
#         self.model = self.get_pretrained_model().to(device)
#         self.optimizer = opt.AdamW(self.model.parameters(), lr=self.args.lr)
#         self.scheduler = opt.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)
#         self.criterion = nn.CrossEntropyLoss()

#     def train(self):
#         print('Checkpoint File Path : {}'.format(self.ckpt_path))
#         train_dataset = TorchDataset(paths=self.ft_paths, sfreq=self.sfreq, rfreq=self.rfreq)
#         train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size,
#                                       shuffle=True, drop_last=True)
#         eval_dataset = TorchDataset(paths=self.eval_paths, sfreq=self.sfreq, rfreq=self.rfreq)
#         eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.args.batch_size, drop_last=False)

#         best_model_state, best_mf1 = None, 0.0
#         best_pred, best_real = [], []

#         for epoch in range(self.args.epochs):
#             self.model.train()
#             epoch_train_loss = []
#             for data in train_dataloader:
#                 self.optimizer.zero_grad()
#                 x, y = data
#                 x, y = x.to(device), y.to(device)

#                 pred = self.model(x)
#                 loss = self.criterion(pred, y)

#                 epoch_train_loss.append(float(loss.detach().cpu().item()))
#                 loss.backward()
#                 self.optimizer.step()

#             self.model.eval()
#             epoch_test_loss = []
#             epoch_real, epoch_pred = [], []
#             for data in eval_dataloader:
#                 with torch.no_grad():
#                     x, y = data
#                     x, y = x.to(device), y.to(device)
#                     pred = self.model(x)
#                     loss = self.criterion(pred, y)
#                     pred = pred.argmax(dim=-1)
#                     real = y

#                     epoch_real.extend(list(real.detach().cpu().numpy()))
#                     epoch_pred.extend(list(pred.detach().cpu().numpy()))
#                     epoch_test_loss.append(float(loss.detach().cpu().item()))

#             epoch_train_loss, epoch_test_loss = np.mean(epoch_train_loss), np.mean(epoch_test_loss)
#             eval_acc, eval_mf1 = accuracy_score(y_true=epoch_real, y_pred=epoch_pred), \
#                                  f1_score(y_true=epoch_real, y_pred=epoch_pred, average='macro')

#             print('[Epoch] : {0:03d} \t '
#                   '[Train Loss] => {1:.4f} \t '
#                   '[Evaluation Loss] => {2:.4f} \t '
#                   '[Evaluation Accuracy] => {3:.4f} \t'
#                   '[Evaluation Macro-F1] => {4:.4f}'.format(epoch + 1, epoch_train_loss, epoch_test_loss,
#                                                             eval_acc, eval_mf1))

#             if best_mf1 < eval_mf1:
#                 best_mf1 = eval_mf1
#                 best_model_state = self.model.state_dict()
#                 best_pred, best_real = epoch_pred, epoch_real

#             self.scheduler.step()

#         self.save_ckpt(best_model_state, best_pred, best_real)

#     def save_ckpt(self, model_state, pred, real):
#         if not os.path.exists(os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'linear_prob')):
#             os.makedirs(os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'linear_prob'))

#         save_path = os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'linear_prob', 'best_model.pth')
#         torch.save({
#             'backbone_name': 'NeuroNet_LinearProb',
#             'model_state': model_state,
#             'hyperparameter': self.args.__dict__,
#             'result': {'real': real, 'pred': pred},
#             'paths': {'train_paths': self.ft_paths, 'eval_paths': self.eval_paths}
#         }, save_path)

#     def get_pretrained_model(self):
#         # 1. Prepared Pretrained Model
#         model_parameter = self.ckpt['model_parameter']
#         pretrained_model = NeuroNet(**model_parameter)
#         pretrained_model.load_state_dict(self.ckpt['model_state'])

#         # 2. Encoder Wrapper
#         backbone = NeuroNetEncoderWrapper(
#             fs=model_parameter['fs'], second=model_parameter['second'],
#             time_window=model_parameter['time_window'], time_step=model_parameter['time_step'],
#             frame_backbone=pretrained_model.frame_backbone,
#             patch_embed=pretrained_model.autoencoder.patch_embed,
#             encoder_block=pretrained_model.autoencoder.encoder_block,
#             encoder_norm=pretrained_model.autoencoder.encoder_norm,
#             cls_token=pretrained_model.autoencoder.cls_token,
#             pos_embed=pretrained_model.autoencoder.pos_embed,
#             final_length=pretrained_model.autoencoder.embed_dim
#         )

#         # 3. Generator Classifier
#         model = Classifier(backbone=backbone,
#                            backbone_final_length=pretrained_model.autoencoder.embed_dim)
#         return model


# class TorchDataset(Dataset):
#     def __init__(self, paths: List, sfreq: int, rfreq: int):
#         self.paths = paths
#         self.info = mne.create_info(sfreq=sfreq, ch_types='eeg', ch_names=['Fp1'])
#         self.xs, self.ys = self.get_data(rfreq)

#     def __len__(self):
#         return self.xs.shape[0]

#     def get_data(self, rfreq):
#         xs, ys = [], []
#         for path in self.paths:
#             data = np.load(path)
#             x, y = data['x'], data['y']
#             x = np.expand_dims(x, axis=1)
#             x = mne.EpochsArray(x, info=self.info)
#             x = x.resample(rfreq)
#             x = x.get_data().squeeze()
#             xs.append(x)
#             ys.append(y)
#         xs = np.concatenate(xs, axis=0)
#         ys = np.concatenate(ys, axis=0)
#         return xs, ys

#     def __getitem__(self, idx):
#         x = torch.tensor(self.xs[idx], dtype=torch.float)
#         y = torch.tensor(self.ys[idx], dtype=torch.long)
#         return x, y


# if __name__ == '__main__':
#     augments = get_args()
#     for n_fold in range(10):
#         augments.n_fold = n_fold
#         trainer = Trainer(augments)
#         trainer.train()

# -*- coding:utf-8 -*-
import os
import sys
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Đảm bảo nạp được các module từ thư mục gốc
sys.path.append(os.getcwd())
from models.neuronet.model import NeuroNet
from models.utils import model_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser()
    # Đường dẫn
    parser.add_argument('--npz_dir', required=True, help="Thư mục chứa file .npz")
    parser.add_argument('--split_json', required=True, help="File JSON chia fold")
    parser.add_argument('--pretrained_path', required=True, help="Đường dẫn file .pth đã pre-train")
    
    # Tham số huấn luyện
    parser.add_argument('--n_fold', default=0, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=512, type=int) # Linear prob có thể dùng batch rất lớn
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--input_channels', default=1, type=int)
    
    parser.add_argument('--ckpt_path', default='ckpt_linear_prob', type=str)
    return parser.parse_args()

class TorchDataset(Dataset):
    def __init__(self, subj_list, npz_dir):
        super().__init__()
        self.x, self.y = [], []
        print(f">>> Loading {len(subj_list)} subjects...")
        for sid in subj_list:
            path = os.path.join(npz_dir, f"{sid}.npz" if not sid.endswith('.npz') else sid)
            data = np.load(path)
            x, y = data['x'], data['y']
            # Chuẩn hóa Z-score
            x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)
            self.x.append(x)
            self.y.append(y)
        self.x = torch.tensor(np.concatenate(self.x), dtype=torch.float32)
        self.y = torch.tensor(np.concatenate(self.y), dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

class LinearClassifier(nn.Module):
    def __init__(self, encoder, embed_dim=256):
        super().__init__()
        self.encoder = encoder
        # ĐÓNG BĂNG ENCODER: Không cho phép cập nhật trọng số
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Chỉ huấn luyện lớp này
        self.fc = nn.Linear(embed_dim, 5)

    def forward(self, x):
        with torch.no_grad(): # Đảm bảo không tính toán gradient cho encoder
            feat = self.encoder.forward_latent(x)
        return self.fc(feat)

class LinearProbTrainer:
    def __init__(self, args):
        self.args = args
        
        # 1. Khởi tạo NeuroNet gốc
        base_model = NeuroNet(
            fs=100, second=30, time_window=3, time_step=0.375,
            encoder_embed_dim=256, encoder_heads=8, encoder_depths=4,
            decoder_embed_dim=128, decoder_heads=4, decoder_depths=2,
            projection_hidden=[512, 256], input_channels=args.input_channels
        )
        
        # 2. Nạp trọng số Pre-trained
        print(f">>> Loading SSL weights: {args.pretrained_path}")
        ckpt = torch.load(args.pretrained_path, map_location='cpu')
        base_model.load_state_dict(ckpt['model_state'])
        
        # 3. Tạo Linear Probing Model
        self.model = LinearClassifier(base_model, embed_dim=256).to(device)
        
        # 4. Optimizer (Chỉ tối ưu lớp FC)
        self.optimizer = opt.AdamW(self.model.fc.parameters(), lr=args.lr)
        
        # 5. Weighted Cross Entropy (Xử lý mất cân bằng lớp)
        # Tính dựa trên thống kê thực tế của bạn
        weights = torch.tensor([1.15, 1.78, 0.55, 1.03, 1.30]).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

    def train(self):
        with open(self.args.split_json, 'r') as f: splits = json.load(f)
        fold = splits[f"fold_{self.args.n_fold}"]
        
        train_loader = DataLoader(TorchDataset(fold['train'], self.args.npz_dir), batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(TorchDataset(fold['test'], self.args.npz_dir), batch_size=self.args.batch_size)

        best_f1 = 0
        for epoch in range(self.args.epochs):
            self.model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()

            acc, f1, report = self.evaluate(test_loader)
            print(f"Epoch {epoch+1} | Acc: {acc*100:.2f}% | F1: {f1*100:.2f}%")
            
            if f1 > best_f1:
                best_f1 = f1
                print(f"🌟 New Best Linear Probe Score!\n{report}")
                save_path = os.path.join(self.args.ckpt_path, f"linear_probe_fold{self.args.n_fold}.pth")
                torch.save(self.model.state_dict(), save_path)

    def evaluate(self, loader):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in loader:
                out = self.model(x.to(device))
                all_preds.append(out.argmax(dim=1).cpu().numpy())
                all_labels.append(y.numpy())
        
        y_true, y_pred = np.concatenate(all_labels), np.concatenate(all_preds)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        report = classification_report(y_true, y_pred, target_names=['W', 'N1', 'N2', 'N3', 'REM'], digits=4)
        return acc, f1, report

if __name__ == '__main__':
    import argparse
    args = get_args()
    os.makedirs(args.ckpt_path, exist_ok=True)
    LinearProbTrainer(args).train()