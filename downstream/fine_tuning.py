# -*- coding:utf-8 -*-
import os, sys, json, torch, argparse, warnings
import numpy as np
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Thêm module Mamba (phải cài: pip install mamba-ssm)
try:
    from mamba_ssm import Mamba
except:
    Mamba = None

sys.path.append(os.getcwd())
from models.neuronet.model import NeuroNet, NeuroNetEncoderWrapper
from pretrained.data_loader import TorchDataset # Chúng ta sẽ dùng bản nâng cấp cho chuỗi

warnings.filterwarnings(action='ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- BỘ NẠP DỮ LIỆU CHUỖI THỜI GIAN (MANY-TO-MANY) ---
class SeqDataset(Dataset):
    def __init__(self, subj_list, npz_dir, context_len=20):
        self.x, self.y = [], []
        print(f">>> Loading {len(subj_list)} subjects for Fine-tuning...")
        for sid in subj_list:
            path = os.path.join(npz_dir, f"{sid}.npz" if not sid.endswith('.npz') else sid)
            data = np.load(path)
            vx, vy = data['x'], data['y']
            # Chuẩn hóa
            vx = (vx - vx.mean(axis=-1, keepdims=True)) / (vx.std(axis=-1, keepdims=True) + 1e-6)
            
            # Cắt Many-to-Many (Chuỗi context_len epoch liên tiếp)
            for i in range(0, len(vy) - context_len, context_len // 2):
                self.x.append(vx[i : i + context_len])
                self.y.append(vy[i : i + context_len])
                
        self.x = torch.tensor(np.array(self.x), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

# --- MODULE HỌC NGỮ CẢNH MAMBA ---
class MambaTCM(nn.Module):
    def __init__(self, backbone, embed_dim=256):
        super().__init__()
        self.backbone = backbone
        # Đóng băng Backbone một phần (giữ tri thức SSL)
        for param in self.backbone.parameters():
            param.requires_grad = True # Có thể chỉnh False nếu muốn đóng băng hoàn toàn
            
        self.mamba = Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
        self.classifier = nn.Linear(embed_dim, 5)

    def forward(self, x):
        # x shape: (batch, seq_len, channels, 3000)
        b, s, c, t = x.shape
        x = x.view(b * s, c, t)
        
        # Trích xuất đặc trưng từng epoch
        feat = self.backbone(x) # (b*s, 256)
        feat = feat.view(b, s, -1) # (b, s, 256)
        
        # Học chuỗi thời gian qua Mamba
        out = self.mamba(feat) # (b, s, 256)
        return self.classifier(out) # (b, s, 5)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_dir', required=True)
    parser.add_argument('--split_json', required=True)
    parser.add_argument('--pretrained_ckpt', required=True, help="File .pth từ bước SSL")
    parser.add_argument('--input_channels', type=int, default=1)
    parser.add_argument('--n_fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    return parser.parse_args()

# --- LUỒNG TRAIN CHÍNH ---
def run_finetune(args):
    # 1. Load Folds
    with open(args.split_json, 'r') as f: splits = json.load(f)
    fold = splits[f"fold_{args.n_fold}"]
    
    # 2. Khởi tạo mô hình & Nạp trọng số SSL
    base_model = NeuroNet(fs=100, second=30, time_window=3, time_step=0.375,
                          encoder_embed_dim=256, encoder_heads=8, encoder_depths=4,
                          decoder_embed_dim=128, decoder_heads=4, decoder_depths=2,
                          projection_hidden=[512, 256], input_channels=args.input_channels).to(device)
    
    print(f">>> Loading SSL Weights: {args.pretrained_ckpt}")
    ckpt = torch.load(args.pretrained_ckpt, map_location=device)
    base_model.load_state_dict(ckpt['model_state'])
    
    # Tạo mô hình Fine-tune với Mamba
    model = MambaTCM(base_model.forward_latent).to(device)
    
    # 3. Optimizer & Loss (Weighted)
    optimizer = opt.AdamW(model.parameters(), lr=args.lr)
    # Trọng số tính từ thống kê của bạn (W, N1, N2, N3, REM)
    weights = torch.tensor([1.15, 1.78, 0.55, 1.03, 1.30]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # 4. Load Data
    train_loader = DataLoader(SeqDataset(fold['train'], args.npz_dir), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(SeqDataset(fold['test'], args.npz_dir), batch_size=args.batch_size)

    # 5. Vòng lặp Train
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x) # (b, s, 5)
            loss = criterion(out.view(-1, 5), y.view(-1))
            loss.backward()
            optimizer.step()
        
        # Đánh giá
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                out = model(x.to(device))
                all_preds.append(out.argmax(dim=-1).cpu().numpy().flatten())
                all_labels.append(y.numpy().flatten())
        
        y_true, y_pred = np.concatenate(all_labels), np.concatenate(all_preds)
        print(f"Epoch {epoch+1} | Acc: {accuracy_score(y_true, y_pred):.4f} | F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
        if (epoch+1) % 10 == 0:
            print(classification_report(y_true, y_pred, target_names=['W', 'N1', 'N2', 'N3', 'REM']))

if __name__ == '__main__':
    run_finetune(get_args())

    
# # -*- coding:utf-8 -*-
# import os
# import mne
# import torch
# import random
# import argparse
# import warnings
# import numpy as np
# import torch.nn as nn
# import torch.optim as opt
# from mamba_ssm import Mamba
# from models.utils import model_size
# from torch.utils.data import Dataset, DataLoader
# from models.neuronet.model import NeuroNet, NeuroNetEncoderWrapper
# from sklearn.metrics import accuracy_score, f1_score


# warnings.filterwarnings(action='ignore')


# random_seed = 777
# np.random.seed(random_seed)
# torch.manual_seed(random_seed)
# random.seed(random_seed)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)


# def get_args():
#     file_name = 'mini'
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--n_fold', default=1, choices=[0, 1, 2, 3, 4])
#     parser.add_argument('--ckpt_path', default=os.path.join('..', '..', '..', 'ckpt',
#                                                             'SHHS', 'cm_eeg', file_name), type=str)
#     parser.add_argument('--temporal_context_length', default=20)
#     parser.add_argument('--window_size', default=10)
#     parser.add_argument('--epochs', default=150, type=int)
#     parser.add_argument('--batch_size', default=64, type=int)
#     parser.add_argument('--lr', default=0.0005, type=float)

#     parser.add_argument('--embed_dim', default=256)
#     parser.add_argument('--temporal_context_modules', choices=['lstm', 'mha', 'lstm_mha', 'mamba'], default='mamba')
#     parser.add_argument('--save_path', default=os.path.join('..', '..', '..',
#                                                             'ckpt', 'SHHS', 'cm_eeg',
#                                                             file_name), type=str)
#     return parser.parse_args()


# class TemporalContextModule(nn.Module):
#     def __init__(self, backbone, backbone_final_length, embed_dim):
#         super().__init__()
#         self.backbone = self.freeze_backbone(backbone)
#         self.backbone_final_length = backbone_final_length
#         self.embed_dim = embed_dim
#         self.embed_layer = nn.Sequential(
#             nn.Linear(backbone_final_length, embed_dim),
#             nn.BatchNorm1d(embed_dim),
#             nn.ELU(),
#             nn.Linear(embed_dim, embed_dim)
#         )

#     def apply_backbone(self, x):
#         out = []
#         for x_ in torch.split(x, dim=1, split_size_or_sections=1):
#             o = self.backbone(x_.squeeze())
#             o = self.embed_layer(o)
#             out.append(o)
#         out = torch.stack(out, dim=1)
#         return out

#     @staticmethod
#     def freeze_backbone(backbone: nn.Module):
#         for name, module in backbone.named_modules():
#             if name in ['encoder_block.3.ls1', 'encoder_block.3.drop_path1', 'encoder_block.3.norm2',
#                         'encoder_block.3.mlp', 'encoder_block.3.mlp.fc1', 'encoder_block.3.mlp.act',
#                         'encoder_block.3.mlp.drop1', 'encoder_block.3.mlp.norm', 'encoder_block.3.mlp.fc2',
#                         'encoder_block.3.mlp.drop2', 'encoder_block.3.ls2', 'encoder_block.3.drop_path2',
#                         'encoder_norm']:
#                 for param in module.parameters():
#                     param.requires_grad = True
#             else:
#                 for param in module.parameters():
#                     param.requires_grad = False
#         return backbone


# class LSTM_TCM(TemporalContextModule):
#     def __init__(self, backbone, backbone_final_length, embed_dim):
#         super().__init__(backbone=backbone, backbone_final_length=backbone_final_length, embed_dim=embed_dim)
#         self.rnn_layer = 2
#         self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.embed_dim, num_layers=self.rnn_layer)
#         self.fc = nn.Linear(self.embed_dim, 5)

#     def forward(self, x):
#         x = self.apply_backbone(x)
#         x, _ = self.lstm(x)
#         x = self.fc(x)
#         return x


# class MHA_TCM(TemporalContextModule):
#     def __init__(self, backbone, backbone_final_length, embed_dim):
#         super().__init__(backbone=backbone, backbone_final_length=backbone_final_length, embed_dim=embed_dim)
#         self.mha_heads = 8
#         self.mha_layer = 2
#         self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.embed_dim, self.mha_heads),
#                                                  num_layers=self.mha_layer)
#         self.fc = nn.Linear(self.embed_dim, 5)

#     def forward(self, x):
#         x = self.apply_backbone(x)
#         x = self.transformer(x)
#         x = self.fc(x)
#         return x


# class LSTM_MHA_TCM(TemporalContextModule):
#     def __init__(self, backbone, backbone_final_length, embed_dim):
#         super().__init__(backbone=backbone, backbone_final_length=backbone_final_length, embed_dim=embed_dim)
#         self.mha_heads = 8
#         self.mha_layer = 2
#         self.rnn_layer = 1
#         self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.embed_dim, num_layers=self.rnn_layer)
#         self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.embed_dim, self.mha_heads),
#                                                  num_layers=self.mha_layer)
#         self.fc = nn.Linear(self.embed_dim, 5)

#     def forward(self, x):
#         x = self.apply_backbone(x)
#         x, _ = self.lstm(x)
#         x = self.transformer(x)
#         x = self.fc(x)
#         return x


# class MAMBA_TCM(TemporalContextModule):
#     def __init__(self, backbone, backbone_final_length, embed_dim):
#         super().__init__(backbone=backbone, backbone_final_length=backbone_final_length, embed_dim=embed_dim)
#         self.mamba_heads = 8
#         self.mamba_layer = 1
#         self.mamba = nn.Sequential(*[
#             Mamba(d_model=self.embed_dim,
#                   d_state=16,
#                   d_conv=4,
#                   expand=2)
#             for _ in range(self.mamba_layer)
#         ])
#         self.fc = nn.Linear(self.embed_dim, 5)

#     def forward(self, x):
#         x = self.apply_backbone(x)
#         x = self.mamba(x)
#         x = self.fc(x)
#         return x


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
#         train_dataset, eval_dataset = TorchDataset(paths=self.ft_paths,
#                                                    temporal_context_length=self.args.temporal_context_length,
#                                                    window_size=self.args.window_size,
#                                                    sfreq=self.sfreq, rfreq=self.rfreq), \
#                                       TorchDataset(paths=self.eval_paths,
#                                                    temporal_context_length=self.args.temporal_context_length,
#                                                    window_size=self.args.temporal_context_length,
#                                                    sfreq=self.sfreq, rfreq=self.rfreq)

#         train_dataloader, eval_dataloader = DataLoader(dataset=train_dataset,
#                                                        batch_size=self.args.batch_size,
#                                                        shuffle=True), \
#                                             DataLoader(dataset=eval_dataset,
#                                                        batch_size=self.args.batch_size,
#                                                        shuffle=False)

#         best_model_state, best_mf1 = None, 0.0
#         best_pred, best_real = [], []

#         for epoch in range(self.args.epochs):
#             self.model.train()
#             epoch_train_loss = []
#             for batch in train_dataloader:
#                 self.optimizer.zero_grad()
#                 x, y = batch
#                 x, y = x.to(device), y.to(device)

#                 out = self.model(x)
#                 loss, pred, real = self.get_loss(out, y)

#                 epoch_train_loss.append(float(loss.detach().cpu().item()))
#                 loss.backward()
#                 self.optimizer.step()

#             self.model.eval()
#             epoch_test_loss = []
#             epoch_real, epoch_pred = [], []
#             for batch in eval_dataloader:
#                 x, y = batch
#                 x, y = x.to(device), y.to(device)
#                 try:
#                     out = self.model(x)
#                 except IndexError:
#                     continue
#                 loss, pred, real = self.get_loss(out, y)
#                 pred = torch.argmax(pred, dim=-1)
#                 epoch_real.extend(list(real.detach().cpu().numpy()))
#                 epoch_pred.extend(list(pred.detach().cpu().numpy()))
#                 epoch_test_loss.append(float(loss.detach().cpu().item()))

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
#         if not os.path.exists(os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'fine_tuning')):
#             os.makedirs(os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'fine_tuning'))

#         save_path = os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'fine_tuning', 'best_model.pth')
#         torch.save({
#             'backbone_name': 'NeuroNet_FineTuning',
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
#             final_length=pretrained_model.autoencoder.embed_dim,
#         )

#         # 3. Temporal Context Module
#         tcm = self.get_temporal_context_module()
#         model = tcm(backbone=backbone,
#                     backbone_final_length=pretrained_model.autoencoder.embed_dim,
#                     embed_dim=self.args.embed_dim)
#         return model

#     def get_temporal_context_module(self):
#         if self.args.temporal_context_modules == 'lstm':
#             return LSTM_TCM
#         if self.args.temporal_context_modules == 'mha':
#             return MHA_TCM
#         if self.args.temporal_context_modules == 'lstm_mha':
#             return LSTM_MHA_TCM
#         if self.args.temporal_context_modules == 'mamba':
#             return MAMBA_TCM

#     def get_loss(self, pred, real):
#         if pred.dim() == 3:
#             pred = pred.view(-1, pred.size(2))
#             real = real.view(-1)
#         loss = self.criterion(pred, real)
#         return loss, pred, real


# class TorchDataset(Dataset):
#     def __init__(self, paths, temporal_context_length, window_size,
#                  sfreq: int, rfreq: int):
#         super().__init__()
#         self.sfreq, self.rfreq = sfreq, rfreq
#         self.info = mne.create_info(sfreq=sfreq, ch_types='eeg', ch_names=['Fp1'])
#         self.x, self.y = self.get_data(paths,
#                                        temporal_context_length=temporal_context_length,
#                                        window_size=window_size)
#         self.x, self.y = torch.tensor(self.x, dtype=torch.float32), torch.tensor(self.y, dtype=torch.long)

#     def get_data(self, paths, temporal_context_length, window_size):
#         total_x, total_y = [], []
#         for path in paths:
#             data = np.load(path)
#             x, y = data['x'], data['y']
#             x = np.expand_dims(x, axis=1)
#             x = mne.EpochsArray(x, info=self.info)
#             x = x.resample(self.rfreq)
#             x = x.get_data().squeeze()
#             x = self.many_to_many(x, temporal_context_length, window_size)
#             y = self.many_to_many(y, temporal_context_length, window_size)
#             total_x.append(x)
#             total_y.append(y)
#         total_x, total_y = np.concatenate(total_x), np.concatenate(total_y)
#         return total_x, total_y

#     @staticmethod
#     def many_to_many(elements, temporal_context_length, window_size):
#         size = len(elements)
#         total = []
#         if size <= temporal_context_length:
#             return elements
#         for i in range(0, size-temporal_context_length+1, window_size):
#             temp = np.array(elements[i:i+temporal_context_length])
#             total.append(temp)
#         total.append(elements[size-temporal_context_length:size])
#         total = np.array(total)
#         return total

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, item):
#         x = torch.tensor(self.x[item])
#         y = torch.tensor(self.y[item])
#         return x, y


# if __name__ == '__main__':
#     augments = get_args()
#     for n_fold in range(10):
#         augments.n_fold = n_fold
#         trainer = Trainer(augments)
#         trainer.train()
