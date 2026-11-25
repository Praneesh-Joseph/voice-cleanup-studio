import os
import numpy as np
import soundfile as sf
from vad import frame_energy_db, detect_speech, silence_cuts, trim_silences

SR = 16000
rng = np.random.default_rng(0)
results = []


def check(name, ok, info=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"  -> {info}"))
    results.append(ok)


def burst(sec):
    t = np.arange(int(sec * SR)) / SR
    y = 0.3 * np.sin(2 * np.pi * 150 * t) + 0.15 * np.sin(2 * np.pi * 300 * t) + 0.08 * np.sin(2 * np.pi * 450 * t)
    return y.astype(np.float32)


def noise(sec):
    return (0.003 * rng.standard_normal(int(sec * SR))).astype(np.float32)


# 1.5 s noise, A 1.0 s, 1.4 s gap, B 0.7 s, 0.3 s gap, C 0.6 s, 1.5 s noise = 7.0 s
x = np.concatenate([noise(1.5), burst(1.0), noise(1.4), burst(0.7), noise(0.3), burst(0.6), noise(1.5)])
before = x.copy()

db = frame_energy_db(np.full(SR, 0.5, np.float32), SR)
check("frame_energy_db shape and level", len(db) == 98 and abs(db.mean() + 6.02) < 0.05, f"{len(db)}, {db.mean():.2f}")

speech = detect_speech(x, SR)
expect = [(1.5, 2.5), (3.9, 4.6), (4.9, 5.5)]
print("      speech found:", [(round(s, 2), round(e, 2)) for s, e in speech])
check("finds the 3 bursts",
      len(speech) == 3 and all(abs(s - a) < 0.15 and abs(e - b) < 0.15 for (s, e), (a, b) in zip(speech, expect)))

cuts = silence_cuts(x, SR)
print("      cuts:", [(round(s, 2), round(e, 2)) for s, e in cuts])
check("3 cuts (start, middle, end)", len(cuts) == 3)
check("first cut starts at 0, last ends at the end", cuts[0][0] == 0.0 and abs(cuts[-1][1] - 7.0) < 1e-6)
check("0.3 s gap is not trimmed", all(not (s < 4.9 and e > 4.6) for s, e in cuts))

out = trim_silences(x, SR)
want = 7.0 - sum(e - s for s, e in cuts) - 0.015  # one internal join loses a 15 ms overlap
check("trimmed length", abs(len(out) / SR - want) < 0.003, f"{len(out) / SR:.3f} vs {want:.3f}")
check("input untouched", np.array_equal(x, before))

quiet = np.zeros(3 * SR, np.float32)
check("digital silence: no speech, audio unchanged",
      detect_speech(quiet, SR) == [] and np.array_equal(trim_silences(quiet, SR), quiet))
hiss = noise(3)
check("noise only: no speech, audio unchanged",
      detect_speech(hiss, SR) == [] and np.array_equal(trim_silences(hiss, SR), hiss))

print("\nall checks passed\n" if all(results) else "\nSOME CHECKS FAILED\n")

# ---- Part 2: your real recording (denoised version from the earlier test) ----
path = "test_data/out/my_fillers_denoised.wav"
if os.path.exists(path):
    y, sr = sf.read(path, dtype="float32")
    segs = detect_speech(y, sr)
    print(f"{path}: {len(y) / sr:.1f} s, {len(segs)} speech segments")
    for s, e in segs:
        print(f"  speech {s:6.2f} -> {e:6.2f}")
    trimmed = trim_silences(y, sr)
    print(f"trimmed: {len(y) / sr:.1f} s -> {len(trimmed) / sr:.1f} s")
    sf.write("test_data/out/my_fillers_trimmed.wav", trimmed, sr)
    print("wrote test_data/out/my_fillers_trimmed.wav")