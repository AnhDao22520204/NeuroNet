# -*- coding:utf-8 -*-
import os, sys, json, torch, argparse, warnings
import numpy as np
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Đảm bảo cài đặt: pip install mamba-ssm causal-conv1d
try:
    from mamba_ssm import Mamba
except ImportError:
    print("❌ Lỗi: Chưa cài đặt mamba-ssm. Hãy chạy '!pip install mamba-ssm causal-conv1d'")
    Mamba = None

# Thêm đường dẫn để nạp module
sys.path.append(os.getcwd())
from models.neuronet.model import NeuroNet, NeuroNetEncoderWrapper

warnings.filterwarnings(action='ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- BỘ NẠP DỮ LIỆU CHUỖI THỜI GIAN (MANY-TO-MANY) ---
class SeqDataset(Dataset):
    def __init__(self, subj_list, npz_dir, context_len=20):
        self.x, self.y = [], []
        print(f">>> Đang chuẩn bị dữ liệu chuỗi ({context_len} epoch) cho {len(subj_list)} bệnh nhân...")
        
        for sid in subj_list:
            path = os.path.join(npz_dir, f"{sid}.npz" if not sid.endswith('.npz') else sid)
            if not os.path.exists(path): continue
            
            data = np.load(path)
            vx, vy = data['x'], data['y']
            
            # 1. Chuẩn hóa Z-score từng epoch
            vx = (vx - vx.mean(axis=-1, keepdims=True)) / (vx.std(axis=-1, keepdims=True) + 1e-6)
            
            # 2. Tạo chuỗi Many-to-Many (trượt cửa sổ)
            # Bước nhảy là context_len // 2 để tăng số lượng mẫu (Data Augmentation)
            for i in range(0, len(vy) - context_len, context_len // 2):
                self.x.append(vx[i : i + context_len])
                self.y.append(vy[i : i + context_len])
        
        self.x = torch.tensor(np.array(self.x), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.long)
        print(f"✅ Tổng cộng có {len(self.y)} chuỗi dữ liệu.")

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

# --- HỆ THỐNG MÔ HÌNH FINE-TUNING (BACKBONE + MAMBA) ---
class MambaTCM(nn.Module):
    def __init__(self, backbone_module, embed_dim=256):
        super().__init__()
        self.backbone = backbone_module
        # Cho phép tinh chỉnh bộ Encoder (unfreeze)
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        # Module Mamba để học ngữ cảnh thời gian
        self.mamba = Mamba(
            d_model=embed_dim, 
            d_state=16, 
            d_conv=4, 
            expand=2
        )
        # Lớp phân loại cuối cùng cho 5 giai đoạn ngủ
        self.classifier = nn.Linear(embed_dim, 5)

    def forward(self, x):
        # x shape: (batch, seq_len, channels, 3000)
        b, s, c, t = x.shape
        
        # 1. Ép chiều để đi qua bộ trích xuất đặc trưng theo từng epoch lẻ
        x = x.view(b * s, c, t) 
        feat = self.backbone(x) # -> (b*s, 256)
        
        # 2. Đưa về dạng chuỗi để Mamba xử lý thời gian
        feat = feat.view(b, s, -1) # -> (b, s, 256)
        
        # 3. Qua Mamba và phân lớp
        out = self.mamba(feat) # -> (b, s, 256)
        return self.classifier(out) # -> (b, s, 5)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_dir', required=True)
    parser.add_argument('--split_json', required=True)
    parser.add_argument('--pretrained_ckpt', required=True, help="File .pth từ bước SSL")
    parser.add_argument('--input_channels', type=int, default=1)
    parser.add_argument('--n_fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16) # Để 16 để tránh lỗi RAM GPU
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--context_len', type=int, default=20)
    parser.add_argument('--save_path', default='ckpt_finetune', type=str)
    return parser.parse_args()

def run_finetune(args):
    # 1. Load Folds
    with open(args.split_json, 'r') as f: splits = json.load(f)
    fold_data = splits[f"fold_{args.n_fold}"]
    
    # 2. Khởi tạo NeuroNet và nạp trọng số SSL
    print(f">>> Đang khởi tạo mô hình với {args.input_channels} kênh...")
    base_model = NeuroNet(
        fs=100, second=30, time_window=3, time_step=0.375,
        encoder_embed_dim=256, encoder_heads=8, encoder_depths=4,
        decoder_embed_dim=128, decoder_heads=4, decoder_depths=2,
        projection_hidden=[512, 256], 
        input_channels=args.input_channels
    ).to(device)
    
    print(f">>> Nạp trọng số SSL từ: {args.pretrained_ckpt}")
    ckpt = torch.load(args.pretrained_ckpt, map_location=device)
    base_model.load_state_dict(ckpt['model_state'])
    
    # 3. Tạo Wrapper và Mamba Model
    backbone_wrapper = NeuroNetEncoderWrapper(
        fs=100, second=30, time_window=3, time_step=0.375,
        frame_backbone=base_model.frame_backbone,
        patch_embed=base_model.autoencoder.patch_embed,
        encoder_block=base_model.autoencoder.encoder_block,
        encoder_norm=base_model.autoencoder.encoder_norm,
        cls_token=base_model.autoencoder.cls_token,
        pos_embed=base_model.autoencoder.pos_embed,
        final_length=256
    )
    
    model = MambaTCM(backbone_wrapper, embed_dim=256).to(device)
    
    # 4. Optimizer & Loss (Weighted)
    optimizer = opt.AdamW(model.parameters(), lr=args.lr)
    # Trọng số xử lý mất cân bằng: W, N1, N2, N3, REM
    weights = torch.tensor([1.15, 1.78, 0.55, 1.03, 1.30]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # 5. Load Data
    train_loader = DataLoader(SeqDataset(fold_data['train'], args.npz_dir, args.context_len), 
                              batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(SeqDataset(fold_data['test'], args.npz_dir, args.context_len), 
                             batch_size=args.batch_size)

    # 6. Vòng lặp huấn luyện
    best_f1 = 0
    os.makedirs(args.save_path, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            out = model(x) # Output shape: (b, s, 5)
            # Ép về dạng (N, 5) để tính CrossEntropy
            loss = criterion(out.view(-1, 5), y.view(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Đánh giá sau mỗi epoch
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                out = model(x.to(device))
                all_preds.append(out.argmax(dim=-1).cpu().numpy().flatten())
                all_labels.append(y.numpy().flatten())
        
        y_true, y_pred = np.concatenate(all_labels), np.concatenate(all_preds)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            save_name = f"best_finetune_fold{args.n_fold}.pth"
            torch.save(model.state_dict(), os.path.join(args.save_path, save_name))
            print(f"🌟 Đã lưu mô hình tốt nhất!")
            if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                print(classification_report(y_true, y_pred, target_names=['W', 'N1', 'N2', 'N3', 'REM'], digits=4))

if __name__ == '__main__':
    run_finetune(get_args())