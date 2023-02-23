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
from modules import load_pretrained_wav2vec

from data import PreprocessDataset

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

    dataset = PreprocessDataset(datadirs, trim_method,
                                sample_rate, preemp, hop_len, win_len, n_fft, n_mels, f_min)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=n_workers)

    wav2vec = load_pretrained_wav2vec(wav2vec_path).to(device)

    speaker_infos = {}
    pbar = tqdm.tqdm(total=len(dataset), ncols=0)
    for _, (speaker_name, audio_path, wav, mel) in enumerate(dataloader):
        if wav.size(-1) < 10:
            continue

        wav = wav.to(device)
        speaker_name = speaker_name[0]
        audio_path = audio_path[0]

        with torch.no_grad():
            feat = wav2vec.extract_features(wav, None)[0]
            feat = feat.detach().cpu().unsqueeze(0)
            mel = mel.squeeze(0)

        fd, tmp_file = mkstemp(suffix=".tar", prefix="utterance-", dir=outdir)
        torch.save({"feat": feat, "mel": mel}, tmp_file)
        os.close(fd)

        if speaker_name not in speaker_infos.keys():
            speaker_infos[speaker_name] = []

        speaker_infos[speaker_name].append(
            {
                "feature_path": Path(tmp_file).name,
                "audio_path": audio_path,
                "feat_len": len(feat),
                "mel_len": len(mel)
            }
        )

        pbar.update(dataloader.batch_size)
    with open(outdir/"metadata.json", "w") as fw:
        json.dump(speaker_infos, fw, indent=2)


if __name__ == "__main__":
    main(**get_args())
