# -*- coding:utf-8 -*-
"""
SSL pre-training for multi-channel NPZ data (JSON splits), aligned with original NeuroNet
hyperparameters: NeuroNet-B, lr = base_lr * batch / 256, temperature 0.05, mask_ratio 0.8,
recon = MAE1 + MAE2, linear probe: PCA(50) + KNN.
"""
import os
import sys
import json
import random
import shutil
import argparse
import warnings

import numpy as np
import torch
import torch.optim as opt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.neuronet.model import NeuroNet
from models.utils import model_size
from pretrained.data_loader import TorchDataset

warnings.filterwarnings(action='ignore')

random_seed = 777


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in ('yes', 'true', 't', '1'):
        return True
    if s in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got {v!r}')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    p = argparse.ArgumentParser(description='NeuroNet SSL (multi-channel NPZ + JSON folds)')
    p.add_argument('--npz_dir', required=True, help='Thư mục chứa file .npz')
    p.add_argument('--split_json', required=True, help='File JSON chia fold (fold_0, train/val/test)')
    p.add_argument('--ckpt_path', default='ckpt', type=str)
    p.add_argument('--model_name', default='NeuroNet_mc', type=str)
    p.add_argument('--n_fold', default=0, type=int)

    p.add_argument('--sfreq', default=100, type=int)
    p.add_argument('--rfreq', default=100, type=int)
    p.add_argument('--second', default=30, type=int)
    p.add_argument('--time_window', default=3, type=int)
    p.add_argument('--time_step', default=0.375, type=float)

    p.add_argument('--train_epochs', default=30, type=int)
    p.add_argument('--train_base_learning_rate', default=1e-5, type=float)
    p.add_argument('--train_batch_size', default=256, type=int)
    p.add_argument('--train_batch_accumulation', default=1, type=int)

    p.add_argument('--encoder_embed_dim', default=768, type=int)
    p.add_argument('--encoder_heads', default=8, type=int)
    p.add_argument('--encoder_depths', default=4, type=int)
    p.add_argument('--decoder_embed_dim', default=256, type=int)
    p.add_argument('--decoder_heads', default=8, type=int)
    p.add_argument('--decoder_depths', default=3, type=int)
    p.add_argument('--projection_hidden', nargs='+', type=int, default=[1024, 512])
    p.add_argument('--temperature', default=0.05, type=float)
    p.add_argument('--alpha', default=1.0, type=float)
    p.add_argument('--mask_ratio', default=0.8, type=float)
    p.add_argument('--input_channels', default=1, type=int, help='1: EEG đơn kênh, 2: EEG+EOG, ...')

    p.add_argument('--zscore_per_epoch', type=str2bool, default=True,
                   help='Z-score theo epoch như TorchDataset update (True = giữ hành vi cũ)')
    p.add_argument('--eval_interval', default=1, type=int, help='Mỗi bao nhiêu epoch chạy linear probe')
    p.add_argument('--num_workers', default=2, type=int)
    p.add_argument('--print_point', default=20, type=int)
    return p.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        set_seed(random_seed)

        self.model = NeuroNet(
            fs=args.rfreq,
            second=args.second,
            time_window=args.time_window,
            time_step=args.time_step,
            encoder_embed_dim=args.encoder_embed_dim,
            encoder_heads=args.encoder_heads,
            encoder_depths=args.encoder_depths,
            decoder_embed_dim=args.decoder_embed_dim,
            decoder_heads=args.decoder_heads,
            decoder_depths=args.decoder_depths,
            projection_hidden=list(args.projection_hidden),
            input_channels=args.input_channels,
            temperature=args.temperature,
        ).to(device)

        self.eff_batch_size = args.train_batch_size * args.train_batch_accumulation
        self.lr = args.train_base_learning_rate * self.eff_batch_size / 256
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = opt.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.train_epochs)

        self.fold_dir = os.path.join(args.ckpt_path, args.model_name, f'fold_{args.n_fold}')
        self.tensorboard_path = os.path.join(self.fold_dir, 'tensorboard')
        if os.path.exists(self.tensorboard_path):
            shutil.rmtree(self.tensorboard_path)
        os.makedirs(self.fold_dir, exist_ok=True)
        self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_path)

        print('Model Size : {:.2f} MB'.format(model_size(self.model)))
        print('Frame (patches) : {}'.format(self.model.num_patches))
        print('Learning rate : {:.2e} (eff_batch={})'.format(self.lr, self.eff_batch_size))

    def train(self):
        with open(self.args.split_json, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        fold = splits[f'fold_{self.args.n_fold}']

        z = self.args.zscore_per_epoch
        train_loader = DataLoader(
            TorchDataset(fold['train'], self.args.npz_dir, z),
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            TorchDataset(fold['val'], self.args.npz_dir, z),
            batch_size=self.args.train_batch_size,
            drop_last=False,
        )
        eval_loader = DataLoader(
            TorchDataset(fold['test'], self.args.npz_dir, z),
            batch_size=self.args.train_batch_size,
            drop_last=False,
        )

        total_step = 0
        best_state = self.model.state_dict()
        best_score = -1.0

        for epoch in range(self.args.train_epochs):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            step = 0

            for x, _ in train_loader:
                x = x.to(device, non_blocking=True)
                out = self.model(x, mask_ratio=self.args.mask_ratio)
                recon_loss, contrastive_loss, (cl_labels, cl_logits) = out
                loss = recon_loss + self.args.alpha * contrastive_loss
                loss.backward()

                if (step + 1) % self.args.train_batch_accumulation == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if (total_step + 1) % self.args.print_point == 0:
                    acc_cl = self.compute_metrics(cl_logits, cl_labels)
                    print(
                        '[Epoch] {:03d} [Step] {:06d} | recon {:.4f} | contrast {:.4f} | '
                        'total {:.4f} | cl_acc {:.4f}'.format(
                            epoch, total_step + 1, recon_loss.item(), contrastive_loss.item(),
                            loss.item(), acc_cl,
                        )
                    )
                self.tensorboard_writer.add_scalar('train/recon', recon_loss.item(), total_step)
                self.tensorboard_writer.add_scalar('train/contrastive', contrastive_loss.item(), total_step)
                self.tensorboard_writer.add_scalar('train/total', loss.item(), total_step)

                step += 1
                total_step += 1

            if (epoch + 1) % self.args.eval_interval == 0 or epoch == 0:
                val_acc, val_mf1 = self.linear_probing(val_loader, eval_loader)
                if val_mf1 > best_score:
                    best_state = self.model.state_dict()
                    best_score = val_mf1
                    self.save_ckpt(best_state, val_mf1, epoch)
                print('[Epoch] {:03d} | val_acc {:.4f} | val_macro_f1 {:.4f}'.format(
                    epoch, val_acc * 100, val_mf1 * 100))
                self.tensorboard_writer.add_scalar('val/accuracy', val_acc, total_step)
                self.tensorboard_writer.add_scalar('val/macro_f1', val_mf1, total_step)

            self.scheduler.step()

    def linear_probing(self, val_dataloader, eval_dataloader):
        self.model.eval()
        train_x, train_y = self.get_latent_vector(val_dataloader)
        test_x, test_y = self.get_latent_vector(eval_dataloader)
        pca = PCA(n_components=50)
        train_x = pca.fit_transform(train_x)
        test_x = pca.transform(test_x)
        knn = KNeighborsClassifier()
        knn.fit(train_x, train_y)
        pred = knn.predict(test_x)
        acc = accuracy_score(test_y, pred)
        mf1 = f1_score(test_y, pred, average='macro')
        self.model.train()
        return acc, mf1

    def get_latent_vector(self, dataloader):
        xs, ys = [], []
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                z = self.model.forward_latent(x)
                xs.append(z.detach().cpu().numpy())
                ys.append(y.numpy())
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    def save_ckpt(self, model_state, score, epoch, name='best_model.pth'):
        path = os.path.join(self.fold_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_name': 'NeuroNet',
            'epoch': epoch,
            'score': score,
            'model_state': model_state,
            'model_parameter': {
                'fs': self.args.rfreq,
                'second': self.args.second,
                'time_window': self.args.time_window,
                'time_step': self.args.time_step,
                'encoder_embed_dim': self.args.encoder_embed_dim,
                'encoder_heads': self.args.encoder_heads,
                'encoder_depths': self.args.encoder_depths,
                'decoder_embed_dim': self.args.decoder_embed_dim,
                'decoder_heads': self.args.decoder_heads,
                'decoder_depths': self.args.decoder_depths,
                'projection_hidden': list(self.args.projection_hidden),
                'temperature': self.args.temperature,
                'input_channels': self.args.input_channels,
            },
            'hyperparameter': vars(self.args),
        }, path)

    @staticmethod
    def compute_metrics(output, target):
        pred = output.argmax(dim=-1)
        return torch.mean((pred == target).float()).item()


if __name__ == '__main__':
    Trainer(get_args()).train()
