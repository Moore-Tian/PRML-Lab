import argparse
import concurrent.futures
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import logging
import random
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from random import shuffle
import librosa
import numpy as np
import torch
import torch.multiprocessing as mp
from loguru import logger
from tqdm import tqdm
import utils
from modules.mel_processing import spectrogram_torch
from rich.progress import track
from scipy.io import wavfile

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

hps = utils.get_hparams_from_file("configs/config.json")
dconfig = du.load_config("configs/diffusion.yaml")
sampling_rate = hps.data.sampling_rate
hop_length = hps.data.hop_length
speech_encoder = hps["model"]["speech_encoder"]


def process_one(filename, hmodel, f0p, device, diff=False, mel_extractor=None):
    wav, sr = librosa.load(filename, sr=sampling_rate)
    audio_norm = torch.FloatTensor(wav)
    audio_norm = audio_norm.unsqueeze(0)
    soft_path = filename + ".soft.pt"
    if not os.path.exists(soft_path):
        wav16k = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        c = hmodel.encoder(wav16k)
        torch.save(c.cpu(), soft_path)
    f0_path = filename + ".f0.npy"
    if not os.path.exists(f0_path):
        f0_predictor = utils.get_f0_predictor(f0p, sampling_rate=sampling_rate, hop_length = hop_length, device=None, threshold=0.05)
        f0, uv = f0_predictor.compute_f0_uv(wav)
        np.save(f0_path, np.asanyarray((f0, uv), dtype=object))

    spec_path = filename.replace(".wav", ".spec.pt")
    if not os.path.exists(spec_path):
        if sr != hps.data.sampling_rate:
            raise ValueError(
                "{} SR doesn't match target {} SR".format(
                    sr, hps.data.sampling_rate
                )
            )
        spec = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        spec = torch.squeeze(spec, 0)
        torch.save(spec, spec_path)

    if diff or hps.model.vol_embedding:
        volume_path = filename + ".vol.npy"
        volume_extractor = utils.Volume_Extractor(hop_length)
        if not os.path.exists(volume_path):
            volume = volume_extractor.extract(audio_norm)
            np.save(volume_path, volume.to('cpu').numpy())

    if diff:
        mel_path = filename + ".mel.npy"
        if not os.path.exists(mel_path) and mel_extractor is not None:
            mel_t = mel_extractor.extract(audio_norm.to(device), sampling_rate)
            mel = mel_t.squeeze().to('cpu').numpy()
            np.save(mel_path, mel)
        aug_mel_path = filename + ".aug_mel.npy"
        aug_vol_path = filename + ".aug_vol.npy"
        max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
        max_shift = min(1, np.log10(1/max_amp))
        log10_vol_shift = random.uniform(-1, max_shift)
        keyshift = random.uniform(-5, 5)
        if mel_extractor is not None:
            aug_mel_t = mel_extractor.extract(audio_norm * (10 ** log10_vol_shift), sampling_rate, keyshift = keyshift)
        aug_mel = aug_mel_t.squeeze().to('cpu').numpy()
        aug_vol = volume_extractor.extract(audio_norm * (10 ** log10_vol_shift))
        if not os.path.exists(aug_mel_path):
            np.save(aug_mel_path,np.asanyarray((aug_mel,keyshift),dtype=object))
        if not os.path.exists(aug_vol_path):
            np.save(aug_vol_path,aug_vol.to('cpu').numpy())


def process_batch(file_chunk, f0p, diff=False, mel_extractor=None, device="cpu"):
    logger.info("Loading speech encoder for content...")
    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0
    if torch.cuda.is_available():
        gpu_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_id}")
    logger.info(f"Rank {rank} uses device {device}")
    hmodel = utils.get_speech_encoder(speech_encoder, device=device)
    logger.info(f"Loaded speech encoder for rank {rank}")
    for filename in tqdm(file_chunk, position = rank):
        process_one(filename, hmodel, f0p, device, diff, mel_extractor)

def parallel_process(filenames, num_processes, f0p, diff, mel_extractor, device):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        for i in range(num_processes):
            start = int(i * len(filenames) / num_processes)
            end = int((i + 1) * len(filenames) / num_processes)
            file_chunk = filenames[start:end]
            tasks.append(executor.submit(process_batch, file_chunk, f0p, diff, mel_extractor, device=device))
        for task in tqdm(tasks, position = 0):
            task.result()


def load_wav(wav_path):
    return librosa.load(wav_path, sr=None)


def trim_wav(wav, top_db=40):
    return librosa.effects.trim(wav, top_db=top_db)


def normalize_peak(wav, threshold=1.0):
    peak = np.abs(wav).max()
    if peak > threshold:
        wav = 0.98 * wav / peak
    return wav


def resample_wav(wav, sr, target_sr):
    return librosa.resample(wav, orig_sr=sr, target_sr=target_sr)


def save_wav_to_path(wav, save_path, sr):
    wavfile.write(
        save_path,
        sr,
        (wav * np.iinfo(np.int16).max).astype(np.int16)
    )


def process(item):
    spkdir, wav_name, args = item
    speaker = spkdir.replace("\\", "/").split("/")[-1]

    wav_path = os.path.join(args.in_dir, speaker, wav_name)
    if os.path.exists(wav_path) and '.wav' in wav_path:
        os.makedirs(os.path.join(args.out_dir2, speaker), exist_ok=True)

        wav, sr = load_wav(wav_path)
        wav, _ = trim_wav(wav)
        wav = normalize_peak(wav)
        resampled_wav = resample_wav(wav, sr, args.sr2)

        if not args.skip_loudnorm:
            resampled_wav /= np.max(np.abs(resampled_wav))

        save_path2 = os.path.join(args.out_dir2, speaker, wav_name)
        save_wav_to_path(resampled_wav, save_path2, args.sr2)


def process_all_speakers():
    process_count = 30 if os.cpu_count() > 60 else (os.cpu_count() - 2 if os.cpu_count() > 4 else 1)
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        for speaker in speakers:
            spk_dir = os.path.join(args.in_dir, speaker)
            if os.path.isdir(spk_dir):
                print(spk_dir)
                futures = [executor.submit(process, (spk_dir, i, args)) for i in os.listdir(spk_dir) if i.endswith("wav")]
                for _ in track(concurrent.futures.as_completed(futures), total=len(futures), description="resampling:"):
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    print(f"CPU count: {cpu_count()}")
    speakers = os.listdir(args.in_dir)
    process_all_speakers()
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default=None)
    parser.add_argument(
        "--in_dir", type=str, default="dataset/44k", help="path to input dir"
    )
    parser.add_argument(
        '--f0_predictor', type=str, default="rmvpe", help='Select F0 predictor, can select crepe,pm,dio,harvest,rmvpe,fcpe|default: pm(note: crepe is original F0 using mean filter)'
    )
    parser.add_argument(
        '--num_processes', type=int, default=1, help='You are advised to set the number of processes to the same as the number of CPU cores'
    )
    args = parser.parse_args()
    f0p = args.f0_predictor
    device = args.device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mel_extractor = None
    filenames = glob(f"{args.in_dir}/*/*.wav", recursive=True)  # [:10]
    shuffle(filenames)
    mp.set_start_method("spawn", force=True)

    num_processes = args.num_processes
    if num_processes == 0:
        num_processes = os.cpu_count()

    parallel_process(filenames, num_processes, f0p, args.use_diff, mel_extractor, device)
