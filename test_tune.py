import numpy as np
import soundfile as sf
from pathlib import Path
from denoise import spectral_subtraction
from evaluate import snr_db, make_test_pair

SR = 16000
speech = [sf.read(f, dtype="float32")[0] for f in sorted(Path("test_data/speech").glob("*.flac"))]
noises = {f.stem: sf.read(f, dtype="float32")[0] for f in sorted(Path("test_data/noise").glob("*.wav"))}

print("alpha  beta | mean SNR gain at 5 dB input")
for alpha in (1.0, 1.5, 2.0, 3.0):
    for beta in (0.02, 0.1):
        gains = []
        for s in speech:
            for n in noises.values():
                clean_p, noisy = make_test_pair(s, n, 5, SR)
                den = spectral_subtraction(noisy, SR, alpha=alpha, beta=beta)
                gains.append(snr_db(clean_p, den) - snr_db(clean_p, noisy))
        print(f"{alpha:5.1f} {beta:5.2f} | {np.mean(gains):+.2f} dB")