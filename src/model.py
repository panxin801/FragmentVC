from typing import Tuple, List, Optional
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from modules import Smoother, Extractor, get_cosine_schedule_with_warmup


class FragmentVC(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()

        self.unet = UnetBlock(d_model)
        self.smoothers = nn.TransformerEncoder(
            Smoother(d_model, 2, 1024), num_layers=3)

        self.mel_linear = nn.Linear(d_model, 80)
        self.post_net = nn.Sequential(
            nn.Conv1d(80, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 80, kernel_size=5, padding=2),
            nn.BatchNorm1d(80),
            nn.Tanh(),
            nn.Dropout(0.5),
        )

    def forward(self,
                srcs: Tensor,
                refs: Tensor,
                src_masks: Optional[Tensor] = None,
                ref_masks: Optional[Tensor] = None,
                ) -> Tuple[Tensor, List[Optional[Tensor]]]:
        out, attns = self.unet(
            srcs, refs, src_masks=src_masks, ref_masks=ref_masks)
        out = self.smoothers(out, src_key_padding_mask=src_masks)
        out = self.mel_linear(out)
        out = out.transpose(1, 0).transpose(2, 1)
        refined = self.post_net(out)
        out += refined
        return out, attns


class UnetBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        self.conv1 = nn.Conv1d(80, d_model, 3, padding=1,
                               padding_mode="replicate")
        self.conv2 = nn.Conv1d(d_model, d_model, 3,
                               padding=1, padding_mode="replicate")
        self.conv3 = nn.Conv1d(d_model, d_model, 3,
                               padding=1, padding_mode="replicate")

        self.prenet = nn.Sequential(
            nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, d_model))
        self.extractor1 = Extractor(d_model, 2, 1024, no_residual=True)
        self.extractor2 = Extractor(d_model, 2, 1024)
        self.extractor3 = Extractor(d_model, 2, 1024)

    def forward(self,
                srcs: Tensor,
                refs: Tensor,
                src_masks: Optional[Tensor] = None,
                ref_masks: Optional[Tensor] = None) -> Tuple[Tensor, List[Optional[Tensor]]]:
        tgt = self.prenet(srcs)
        tgt = tgt.transpose(0, 1)

        ref1 = self.conv1(refs)
        ref2 = self.conv2(F.relu(ref1))
        ref3 = self.conv3(F.relu(ref2))

        out, attn1 = self.extractor1(tgt,
                                     ref3.transpose(1, 2).transpose(0, 1),
                                     tgt_key_padding_mask=src_masks,
                                     memory_key_padding_mask=ref_masks)
        out, attn2 = self.extractor2(out,
                                     ref2.transpose(1, 2).transpose(0, 1),
                                     tgt_key_padding_mask=src_masks,
                                     memory_key_padding_mask=ref_masks)
        out, attn3 = self.extractor3(out,
                                     ref3.transpose(1, 2).transpose(0, 1),
                                     tgt_key_padding_mask=src_masks,
                                     memory_key_padding_mask=ref_masks)
        return out, [attn1, attn2, attn3]
