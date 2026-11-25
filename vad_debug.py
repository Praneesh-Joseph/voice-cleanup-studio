import sys
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vad import frame_energy_db, detect_speech

path = sys.argv[1] if len(sys.argv) > 1 else "test_data/out/my_fillers_denoised.wav"
level = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
y, sr = sf.read(path, dtype="float32")
db = frame_energy_db(y, sr)
t = np.arange(len(db)) * 0.010
lo, hi = np.percentile(db, 10), np.percentile(db, 95)
thr = lo + level * (hi - lo)
segs = detect_speech(y, sr, level=level)
print(f"noise floor {lo:.1f} dB, speech level {hi:.1f} dB, threshold {thr:.1f} dB, {len(segs)} segments")

fig, ax = plt.subplots(figsize=(24, 4))
ax.plot(t, db, lw=0.8)
ax.axhline(thr, color="red", lw=1)
for s, e in segs:
    ax.axvspan(s, e, color="lime", alpha=0.25)
ax.set_xlabel("time (s)")
ax.set_ylabel("dB")
plt.tight_layout()
plt.savefig("test_data/out/vad_debug.png", dpi=80)
print("wrote test_data/out/vad_debug.png")