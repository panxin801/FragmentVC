import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Dropout, LayerNorm, Conv1d, MultiheadAttention

from fairseq.models.wav2vec import Wav2Vec2Model
from typing import Tuple, Optional


def load_pretrained_wav2vec(ckpt_path):
    ckpt = torch.load(ckpt_path)
    model = Wav2Vec2Model.build_model(ckpt["args"], task=None)
    model.load_state_dict(ckpt["model"])
    model.remove_pretraining_modules()
    model.eval()
    return model


class Smoother(nn.Module):
    def __init__(self):
        super().__init__()


class Extractor(nn.Module):
    def __init__(self):
        super().__init__()
