import numpy as np
from denoise import spectral_subtraction

sr = 16000
rng = np.random.default_rng(0)
t = np.arange(sr * 3) / sr
noise = 0.1 * rng.standard_normal(len(t))
tone = 0.5 * np.sin(2 * np.pi * 440 * t)

# 0-0.5 s noise only | 0.5-2 s tone + noise | 2-3 s noise only
signal = np.zeros(len(t))
signal[int(0.5 * sr):int(2.0 * sr)] = tone[int(0.5 * sr):int(2.0 * sr)]
x = (signal + noise).astype(np.float32)

y = spectral_subtraction(x, sr)

def rms(a): return np.sqrt(np.mean(a ** 2))

tail = slice(int(2.1 * sr), int(2.9 * sr))      # noise-only region
mid = slice(int(0.7 * sr), int(1.8 * sr))       # tone region

print("noise-only region RMS  before:", rms(x[tail]), " after:", rms(y[tail]))
print("tone region RMS        before:", rms(x[mid]), " after:", rms(y[mid]))
print("length same:", len(x) == len(y))