import torch
import sox
from pathlib import Path
from copy import deepcopy
from librosa.util import find_files
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import random

from .utils import load_wav, log_mel_spectrogram
from ipdb import set_trace


class PreprocessDataset(Dataset):
    def __init__(self, datadirs, trim_method,
                 sample_rate, preemp, hop_len, win_len, n_fft, n_mels, f_min):
        data = []
        for datadir in datadirs:
            datadir = Path(datadir)
            speaker_dirs = [x for x in datadir.iterdir() if x.is_dir()]

            for speaker_dir in speaker_dirs:
                audio_paths = find_files(speaker_dir)
                if len(audio_paths) == 0:
                    continue

                speaker_name = speaker_dir.name
                for audio_path in audio_paths:
                    data.append((speaker_name, audio_path))
        self.trim_method = trim_method
        self.sample_rate = sample_rate
        self.preemp = preemp
        self.hop_len = hop_len
        self.win_len = win_len
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.f_min = f_min
        self.data = data

        if trim_method == "vad":
            tfm = sox.Transformer()
            tfm.vad(location=1)
            tfm.vad(location=-1)
            self.sox_transform = tfm

    def __getitem__(self, index):
        speaker_name, audio_path = self.data[index]
        if self.trim_method == "vad":
            wav = load_wav(audio_path, self.sample_rate, trim=False)
            trim_wav = self.sox_transform.build_array(
                input_array=wav, sample_rate_in=self.sample_rate)
            wav = deepcopy(trim_wav if len(trim_wav) > 10 else wav)
        elif self.trim_method == "librosa":
            wav = load_wav(audio_path, self.sample_rate, trim=True)

        mel = log_mel_spectrogram(
            wav,
            self.preemp,
            self.sample_rate,
            self.n_mels,
            self.n_fft,
            self.hop_len,
            self.win_len,
            self.f_min
        )
        return speaker_name, audio_path, torch.FloatTensor(wav), torch.FloatTensor(mel)

    def __len__(self):
        return len(self.data)


class IntraSpeakerDataset(Dataset):
    def __init__(self, feat_dir, metadata_path, n_samples=5, preload=False) -> None:
        with open(metadata_path, "rt", encoding="utf8") as f:
            metadata = json.load(f)

        executor = ThreadPoolExecutor(max_workers=4)
        futures = []
        for spk_name, utts in metadata.items():
            for utt in utts:
                futures.append(executor.submit(
                    _process_data,
                    spk_name,
                    feat_dir,
                    utt["feature_path"],
                    preload))

        self.data = []
        self.spk_to_idx = {}
        for idx, future in enumerate(tqdm(futures, ncols=0)):
            result = future.result()
            self.data.append(result)
            spk_name = result[0]
            if not spk_name in self.spk_to_idx:
                self.spk_to_idx[spk_name] = [idx]
            else:
                self.spk_to_idx[spk_name].append(idx)

        self.feat_dir = Path(feat_dir)
        self.n_samples = n_samples
        self.preload = preload

    def __len__(self):
        return len(self.data)

    def _get_data(self, idx):
        if self.preload:
            spk_name, feat, mel = self.data[idx]
        else:
            spk_name, feat, mel = _load_data(
                *self.data[idx])  # *args is a collect params
        return spk_name, feat, mel

    def __getitem__(self, idx):
        spk_name, feat, tgt_mel = self._get_data(idx)
        utt_idxs = self.spk_to_idx[spk_name].copy()  # 应该是为了vc对应转换用
        utt_idxs.remove(idx)

        sampled_mels = []
        for sampled_id in random.sample(utt_idxs, self.n_samples):
            sampled_mel = self._get_data(sampled_id)[2]
            sampled_mels.append(sampled_mel)

        ref_mels = torch.cat(sampled_mels, dim=0)
        return feat, ref_mels, tgt_mel


def _process_data(spk_name, feat_dir, feature_path, preload):
    if preload:
        return _load_data(spk_name, feat_dir, feature_path)
    else:
        return spk_name, feat_dir, feature_path


def _load_data(spk_name, feat_dir, feature_path):
    feature = torch.load(Path(feat_dir, feature_path))
    # The same as
    # feature = torch.load(Path(feat_dir)/feature_path)
    feat = feature["feat"]
    mel = feature["mel"]
    return spk_name, feat, mel


def collate_batch(batch):
    set_trace()
    feats, refs, tgts = zip(*batch)

    feat_lens = [len(feat) for feat in feats]
    ref_lens = [len(ref) for ref in refs]
    tgt_lens = [len(tgt) for tgt in tgts]
    overlap_lens = [min(feat_len, tgt_len)
                    for feat_len, tgt_len in zip(feat_lens, tgt_lens)]

    feats = pad_sequence(feats, batch_first=True)

    feat_masks = [torch.arange(feats.size(1)) >=
                  feat_len for feat_len in feat_lens]
    feat_masks = torch.stack(feat_masks)

    refs = pad_sequence(refs, batch_first=True,
                        padding_value=-20)  # Why -20 here
    refs = refs.transpose(1, 2)

    ref_masks = [torch.arange(refs.size(2)) >= ref_len for ref_len in ref_lens]
    ref_masks = torch.stack(ref_masks)

    tgts = pad_sequence(tgts, batch_first=True, padding_value=-20)
    tgts = tgts.transpose(1, 2)

    tgt_masks = [torch.arange(tgts.size(2)) >= tgt_len for tgt_len in tgt_lens]
    tgt_masks = torch.stack(tgt_masks)

    return feats, feat_masks, refs, ref_masks, tgts, tgt_masks, overlap_lens
