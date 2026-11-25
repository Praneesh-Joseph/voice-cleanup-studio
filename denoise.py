import numpy as np
from scipy.signal import stft, istft

N_FFT = 512
OVERLAP = 384  # 75% overlap, hop = 128 samples


def _stft(x, sr):
    """Audio -> complex spectrogram of shape (freq_bins, frames)."""
    _, _, Z = stft(x, fs=sr, window="hann", nperseg=N_FFT, noverlap=OVERLAP)
    return Z


def _istft(Z, sr, length):
    """Complex spectrogram -> audio, trimmed or padded to `length` samples."""
    _, y = istft(Z, fs=sr, window="hann", nperseg=N_FFT, noverlap=OVERLAP)
    y = y[:length]
    if len(y) < length:
        y = np.pad(y, (0, length - len(y)))
    return y.astype(np.float32)


# def spectral_subtraction(x, sr, alpha=2.0, beta=0.02, noise_sec=0.5):
#     """STUB: returns input unchanged until the real version exists."""
#     return x.copy()

def spectral_subtraction(x, sr, alpha=2.0, beta=0.02, noise_sec=0.5):
    x = np.asarray(x, dtype=np.float32)
    Z = _stft(x, sr)
    mag = np.abs(Z)
    phase = np.angle(Z)

    # 1. noise fingerprint: average magnitude over the first noise_sec seconds
    hop = N_FFT - OVERLAP
    n_noise = int(noise_sec * sr / hop)
    n_noise = max(1, min(n_noise, mag.shape[1]))
    noise_mag = mag[:, :n_noise].mean(axis=1, keepdims=True)

    # 2. subtract alpha * noise, but never go below beta * noise (the floor)
    clean_mag = np.maximum(mag - alpha * noise_mag, beta * noise_mag)

    # 3. rebuild with the ORIGINAL phase
    Z_clean = clean_mag * np.exp(1j * phase)
    return _istft(Z_clean, sr, len(x))


def wiener_filter(x, sr, noise_sec=0.5):
    """STUB: returns input unchanged until the real version exists."""
    return x.copy()
