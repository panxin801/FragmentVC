import os
import torch
import logging
import tqdm
import json
from pathlib import Path
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from jsonargparse import ArgumentParser, ActionConfigFile
from tempfile import mkstemp

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')


def get_args():
    parse = ArgumentParser()
    parse.add_argument("datadirs", type=str, nargs="+")
    parse.add_argument("wav2vec_path", type=str)
    parse.add_argument("outdir", type=str)
    parse.add_argument(
        "--trim-method", choices=["librosa", "vad"], default="vad")
    parse.add_argument("--n-workers", type=int, default=cpu_count())
    parse.add_argument("--sample-rate", type=int, default=16000)
    parse.add_argument("--preemp", type=float, default=0.97)
    parse.add_argument("--hop-len", type=int, default=326)
    parse.add_argument("--win-len", type=int, default=1304)
    parse.add_argument("--n-fft", type=int, default=1304)
    parse.add_argument("--n-mels", type=int, default=80)
    parse.add_argument("--f-min", type=int, default=50)
    parse.add_argument("--f-max", type=int, default=8000)
    parse.add_argument("--audio-config", action=ActionConfigFile)
    return vars(parse.parse_args())


def main(datadirs, wav2vec_path, outdir,
         trim_method, n_workers, sample_rate, preemp, hop_len, win_len, n_fft, n_mels, f_min, f_max, audio_config,
         **kwargs):
    outdir = Path(outdir)
    if outdir.exists():
        assert outdir.is_dir()
    else:
        outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main(**get_args())
