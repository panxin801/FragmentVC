import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Dropout, LayerNorm, Conv1d, MultiheadAttention
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from fairseq.models.wav2vec import Wav2Vec2Model
from typing import Tuple, Optional


def load_pretrained_wav2vec(ckpt_path):
    ckpt = torch.load(ckpt_path)
    model = Wav2Vec2Model.build_model(ckpt["args"], task=None)
    model.load_state_dict(ckpt["model"])
    model.remove_pretraining_modules()
    model.eval()
    return model


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1):
    def lr_lambda(cur_step):
        if cur_step < num_warmup_steps:
            return float(cur_step)/float(max(1, num_warmup_steps))
        progress = float(cur_step-num_warmup_steps) / \
            float(max(1, num_training_steps-num_warmup_steps))
        return max(0.0, 0.5*(1.0+math.cos(math.pi*float(num_cycles)*2.0*progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class Smoother(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_hid: int, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)

        self.conv1 = Conv1d(d_model, d_hid, 9, padding=4)
        self.conv2 = Conv1d(d_hid, d_model, 1, padding=0)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src2 = self.self_attn(src, src, src,
                              attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]

        # add and norm
        src += self.dropout1(src2)
        src = self.norm1(src)

        # conv1d
        src2 = src.transpose(0, 1).transpose(1, 2)
        src2 = self.conv2(F.relu(self.conv1(src2)))
        src2 = src2.transpose(1, 2).transpose(0, 1)

        # add and norm
        src += self.dropout2(src2)
        src = self.norm2(src)
        return src


class Extractor(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_hid: int, dropout=0.1, no_residual=False):
        super().__init__()

        self.self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.cross_attn = MultiheadAttention(d_model, n_head, dropout=dropout)

        self.conv1 = Conv1d(d_model, d_hid, 9, padding=4)
        self.conv2 = Conv1d(d_model, d_hid, 1, padding=0)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.no_residual = no_residual

    def forward(self,
                tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None, memory_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Multihead attn
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # add and norm
        tgt += self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # multihead cross attn
        tgt2, attn = self.cross_attn(
            tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)

        # add and norm
        if self.no_residual:
            tgt = self.dropout2(tgt2)
        else:
            tgt += self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # conv1d
        tgt2 = tgt.transpose(0, 1).transpose(1, 2)
        tgt2 = self.conv2(F.relu(self.conv1(tgt2)))
        tgt2 = tgt2.transpose(1, 2).transpose(0, 1)

        # add amd norm
        tgt += self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, attn
