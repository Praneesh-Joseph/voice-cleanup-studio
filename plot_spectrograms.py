import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from denoise import _stft

names = ["clean", "noisy", "denoised"]
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, sharey=True)
ref = None
for ax, name in zip(axes, names):
    x, sr = sf.read(f"test_data/out/{name}_5dB.wav", dtype="float32")
    db = 20 * np.log10(np.abs(_stft(x, sr)) + 1e-8)
    if ref is None:
        ref = db.max()              # same colour scale for all three panels
    ax.imshow(db - ref, origin="lower", aspect="auto", cmap="magma",
              vmin=-80, vmax=0, extent=[0, len(x) / sr, 0, sr / 2])
    ax.set_title(name + " (5 dB)")
    ax.set_ylabel("Hz")
axes[-1].set_xlabel("seconds")
plt.tight_layout()
plt.savefig("test_data/out/spectrograms.png", dpi=120)
plt.show()