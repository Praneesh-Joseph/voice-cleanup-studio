import sys
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from faster_whisper.audio import decode_audio
from denoise import spectral_subtraction
from transcribe import transcribe_words, gap_candidates, _voiced_fraction

SR = 16000
x = decode_audio(sys.argv[1], sampling_rate=SR)

# Use the quietest 0.5 s of the recording as the noise fingerprint
n = int(0.5 * SR)
c = np.cumsum(np.r_[0.0, x.astype(np.float64) ** 2])
i = int(np.argmin((c[n:] - c[:-n]) / n))
noise = x[i:i + n]
print(f"noise fingerprint taken from {i / SR:.2f}-{(i + n) / SR:.2f} s")
y = spectral_subtraction(np.concatenate([noise, x]), SR)[n:].astype(np.float32)
sf.write("test_data/out/my_fillers_denoised.wav", y, SR)

words = transcribe_words(y, SR, model_size="small")
print(" ".join(w["word"] for w in words), "\n")

cands = gap_candidates(y, SR, words)
rows = [(s, e, _voiced_fraction(y[int(s * SR):int(e * SR)], SR)) for s, e in cands]
print("start   end    len    voiced")
for s, e, vf in rows:
    print(f"{s:6.2f} {e:6.2f} {(e - s) * 1000:5.0f} ms  {vf:.2f}  {'KEEP' if vf >= 0.6 else 'drop'}")

keep = [(s, e) for s, e, vf in rows if vf >= 0.6]
pause = np.zeros(int(0.3 * SR), dtype=np.float32)
pieces = []
for s, e in keep:
    pieces += [y[int(s * SR):int(e * SR)], pause]
if pieces:
    sf.write("test_data/out/gap_fillers_only2.wav", np.concatenate(pieces), SR)

# Debug picture: spectrogram, word boundaries (white), kept (green) and dropped (red) candidates
fig, ax = plt.subplots(figsize=(24, 5))
ax.specgram(y, NFFT=512, Fs=SR, noverlap=384, cmap="magma", vmin=-110)
ax.set_ylim(0, 4000)
for w in words:
    ax.axvline(w["start"], color="white", lw=0.5)
for s, e, vf in rows:
    ax.axvspan(s, e, color="lime" if vf >= 0.6 else "red", alpha=0.35)
ax.set_xlabel("time (s)")
plt.tight_layout()
plt.savefig("test_data/out/gap_debug.png", dpi=80)
print("\nwrote my_fillers_denoised.wav, gap_fillers_only2.wav, gap_debug.png in test_data/out")