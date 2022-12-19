import torch
import sox
from pathlib import Path
from copy import deepcopy
from librosa.util import find_files
from torch.utils.data import Dataset
from .utils import load_wav, log_mel_spectrogram


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
