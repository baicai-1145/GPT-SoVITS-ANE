import librosa
import numpy as np
import torch
import torchaudio


def load_audio_tensor(path: str):
    try:
        return torchaudio.load(path)
    except (ImportError, OSError, RuntimeError) as exc:
        if "torchcodec" not in str(exc).lower():
            raise

    audio, sample_rate = librosa.load(path, sr=None, mono=False)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    return torch.from_numpy(np.ascontiguousarray(audio)).float(), int(sample_rate)
