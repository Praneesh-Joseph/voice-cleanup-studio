import numpy as np
import soundfile as sf
from pathlib import Path
from denoise import spectral_subtraction
from evaluate import snr_db, make_test_pair

SR = 16000
OUT = Path("test_data") / "out"
OUT.mkdir(parents=True, exist_ok=True)

speech_file = sorted(Path("test_data/speech").glob("*.flac"))[0]
clean, _ = sf.read(speech_file, dtype="float32")
noise, _ = sf.read("test_data/noise/kitchen.wav", dtype="float32")


def save(name, x):
    peak = np.max(np.abs(x))
    if peak > 1.0:          # avoid clipping in the saved file
        x = x / peak
    sf.write(OUT / name, x, SR)


print(f"speech: {speech_file.name} | noise: kitchen")
print("target | noisy SNR | denoised SNR | improvement")
for target in (0, 5, 10):
    clean_p, noisy = make_test_pair(clean, noise, target, SR)
    den = spectral_subtraction(noisy, SR)
    before, after = snr_db(clean_p, noisy), snr_db(clean_p, den)
    print(f"{target:>4} dB | {before:8.2f} | {after:11.2f} | {after - before:+.2f} dB")
    save(f"clean_{target}dB.wav", clean_p)
    save(f"noisy_{target}dB.wav", noisy)
    save(f"denoised_{target}dB.wav", den)

print("Listen to the files in", OUT)