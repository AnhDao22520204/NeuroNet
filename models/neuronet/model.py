# -*- coding:utf-8 -*-
"""
NeuroNet SSL model aligned with the original NeuroNet repo (MAE + NT-Xent),
with multi-channel input: tensor shape (batch, C, fs * second) e.g. (B, 2, 3000).
"""
import torch
import numpy as np
import torch.nn as nn
from typing import List
from models.neuronet.resnet1d import FrameBackBone
from timm.models.vision_transformer import Block
from models.utils import get_2d_sincos_pos_embed_flexible
from models.loss import NTXentLoss
from functools import partial


class NeuroNet(nn.Module):
    def __init__(self, fs: int, second: int, time_window: int, time_step: float,
                 encoder_embed_dim: int, encoder_heads: int, encoder_depths: int,
                 decoder_embed_dim: int, decoder_heads: int, decoder_depths: int,
                 projection_hidden: List, input_channels: int = 1, temperature: float = 0.05):
        super().__init__()
        self.fs, self.second = fs, second
        self.time_window = time_window
        self.time_step = time_step

        self.num_patches, _ = frame_size(
            fs=fs, second=second, time_window=time_window, time_step=time_step
        )
        self.frame_backbone = FrameBackBone(
            fs=self.fs, window=self.time_window, input_channel=input_channels
        )
        self.autoencoder = MaskedAutoEncoderViT(
            input_size=self.frame_backbone.feature_num,
            encoder_embed_dim=encoder_embed_dim,
            num_patches=self.num_patches,
            encoder_heads=encoder_heads,
            encoder_depths=encoder_depths,
            decoder_embed_dim=decoder_embed_dim,
            decoder_heads=decoder_heads,
            decoder_depths=decoder_depths,
        )
        self.contrastive_loss = NTXentLoss(temperature=temperature)

        projection_hidden = [encoder_embed_dim] + projection_hidden
        projectors = []
        for i, (h1, h2) in enumerate(zip(projection_hidden[:-1], projection_hidden[1:])):
            if i != len(projection_hidden) - 2:
                projectors.append(nn.Linear(h1, h2))
                projectors.append(nn.BatchNorm1d(h2))
                projectors.append(nn.ELU())
            else:
                projectors.append(nn.Linear(h1, h2))
        self.projectors = nn.Sequential(*projectors)
        self.projectors_bn = nn.BatchNorm1d(projection_hidden[-1], affine=False)
        self.norm_pix_loss = False

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.5):
        x = self.make_frame(x)
        x = self.frame_backbone(x)

        latent1, pred1, mask1 = self.autoencoder(x, mask_ratio)
        latent2, pred2, mask2 = self.autoencoder(x, mask_ratio)
        o1, o2 = latent1[:, 0, :], latent2[:, 0, :]
        recon_loss1 = self.forward_mae_loss(x, pred1, mask1)
        recon_loss2 = self.forward_mae_loss(x, pred2, mask2)
        recon_loss = recon_loss1 + recon_loss2

        o1, o2 = self.projectors(o1), self.projectors(o2)
        contrastive_loss, (labels, logits) = self.contrastive_loss(o1, o2)
        return recon_loss, contrastive_loss, (labels, logits)

    def forward_latent(self, x: torch.Tensor):
        x = self.make_frame(x)
        x = self.frame_backbone(x)
        latent = self.autoencoder.forward_encoder(x, mask_ratio=0)[0]
        return latent[:, 0, :]

    def forward_mae_loss(self, real: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor):
        if self.norm_pix_loss:
            mean = real.mean(dim=-1, keepdim=True)
            var = real.var(dim=-1, keepdim=True)
            real = (real - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - real) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def make_frame(self, x: torch.Tensor):
        """(batch, C, T) -> (batch, n_frames, C, window)."""
        size = self.fs * self.second
        step = int(self.time_step * self.fs)
        window = int(self.time_window * self.fs)
        frame = []
        for i in range(0, size, step):
            start_idx, end_idx = i, i + window
            sample = x[..., start_idx:end_idx]
            if sample.shape[-1] == window:
                frame.append(sample)
        return torch.stack(frame, dim=1)


class MaskedAutoEncoderViT(nn.Module):
    def __init__(self, input_size: int, num_patches: int,
                 encoder_embed_dim: int, encoder_heads: int, encoder_depths: int,
                 decoder_embed_dim: int, decoder_heads: int, decoder_depths: int):
        super().__init__()
        self.patch_embed = nn.Linear(input_size, encoder_embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.embed_dim = encoder_embed_dim
        self.encoder_depths = encoder_depths
        self.mlp_ratio = 4.0

        self.input_size = (num_patches, encoder_embed_dim)
        self.patch_size = (1, encoder_embed_dim)
        self.grid_h = int(self.input_size[0] // self.patch_size[0])
        self.grid_w = int(self.input_size[1] // self.patch_size[1])
        self.num_patches = self.grid_h * self.grid_w

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, encoder_embed_dim), requires_grad=False)
        self.encoder_block = nn.ModuleList([
            Block(encoder_embed_dim, encoder_heads, self.mlp_ratio, qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(encoder_depths)
        ])
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim, eps=1e-6)

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, decoder_embed_dim), requires_grad=False)
        self.decoder_block = nn.ModuleList([
            Block(decoder_embed_dim, decoder_heads, self.mlp_ratio, qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(decoder_depths)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        self.decoder_pred = nn.Linear(decoder_embed_dim, input_size, bias=True)
        self.initialize_weights()

    def forward(self, x, mask_ratio=0.8):
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        return latent, pred, mask

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float = 0.5):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.encoder_block:
            x = block(x)
        x = self.encoder_norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore: torch.Tensor):
        x = self.decoder_embed(x[:, 1:, :])
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = x + self.decoder_pos_embed
        for block in self.decoder_block:
            x = block(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x

    @staticmethod
    def random_masking(x, mask_ratio):
        n, l, d = x.shape
        len_keep = int(l * (1 - mask_ratio))
        noise = torch.rand(n, l, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))
        mask = torch.ones([n, l], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed_flexible(
            self.pos_embed.shape[-1], (self.grid_h, self.grid_w), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed_flexible(
            self.decoder_pos_embed.shape[-1], (self.grid_h, self.grid_w), cls_token=False
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def frame_size(fs, second, time_window, time_step):
    x = np.random.randn(1, fs * second)
    size = fs * second
    step = int(time_step * fs)
    window = int(time_window * fs)
    frame = []
    for i in range(0, size, step):
        start_idx, end_idx = i, i + window
        sample = x[..., start_idx: end_idx]
        if sample.shape[-1] == window:
            frame.append(sample)
    frame = np.stack(frame, axis=1)
    return frame.shape[1], frame.shape[2]


class NeuroNetEncoderWrapper(nn.Module):
    def __init__(self, fs: int, second: int, time_window: int, time_step: float,
                 frame_backbone, patch_embed, encoder_block, encoder_norm, cls_token, pos_embed,
                 final_length):
        super().__init__()
        self.fs, self.second = fs, second
        self.time_window = time_window
        self.time_step = time_step
        self.patch_embed = patch_embed
        self.frame_backbone = frame_backbone
        self.encoder_block = encoder_block
        self.encoder_norm = encoder_norm
        self.cls_token = cls_token
        self.pos_embed = pos_embed
        self.final_length = final_length

    def forward(self, x, semantic_token=True):
        x = self.make_frame(x)
        x = self.frame_backbone(x)
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.encoder_block:
            x = block(x)
        x = self.encoder_norm(x)
        x = torch.mean(x[:, 1:, :], dim=1)
        return x

    def make_frame(self, x):
        size = self.fs * self.second
        step = int(self.time_step * self.fs)
        window = int(self.time_window * self.fs)
        frame = []
        for i in range(0, size, step):
            start_idx, end_idx = i, i + window
            sample = x[..., start_idx: end_idx]
            if sample.shape[-1] == window:
                frame.append(sample)
        return torch.stack(frame, dim=1)


if __name__ == '__main__':
    x0 = torch.randn((8, 2, 3000))
    m0 = NeuroNet(
        fs=100, second=30, time_window=3, time_step=0.375,
        encoder_embed_dim=768, encoder_depths=4, encoder_heads=8,
        decoder_embed_dim=256, decoder_heads=8, decoder_depths=3,
        projection_hidden=[1024, 512], input_channels=2, temperature=0.05,
    )
    m0.forward(x0, mask_ratio=0.5)
