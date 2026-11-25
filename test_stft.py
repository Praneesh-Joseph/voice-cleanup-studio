import numpy as np
from denoise import _stft, _istft

sr = 16000
t = np.arange(sr * 2) / sr  # 2 seconds
x = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.randn(len(t))).astype(np.float32)

Z = _stft(x, sr)
y = _istft(Z, sr, len(x))

print("input length :", len(x))
print("output length:", len(y))
print("spectrogram shape (freq bins, frames):", Z.shape)
print("max abs error:", np.max(np.abs(x - y)))