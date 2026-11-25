import numpy as np
from loudness import rms_normalize, rms_dbfs, measure_lufs

sr = 16000
t = np.arange(sr * 2) / sr
quiet = (0.01 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
ok = []

def check(name, cond):
    ok.append(cond); print(("PASS " if cond else "FAIL ") + name)

orig = quiet.copy()
y = rms_normalize(quiet, sr)
check("input not mutated", np.array_equal(quiet, orig))
check("float32 and same length", y.dtype == np.float32 and len(y) == len(quiet))
check("hits -20 dBFS", abs(rms_dbfs(y) - (-20.0)) < 0.05)

spiky = quiet.copy(); spiky[100] = 0.9
y2 = rms_normalize(spiky, sr, target_dbfs=-10.0)
check("peak limit respected", np.max(np.abs(y2)) <= 0.97 + 1e-6)

z = np.zeros(sr, dtype=np.float32)
check("silence unchanged", np.array_equal(rms_normalize(z, sr), z))
check("empty input ok", len(rms_normalize(np.zeros(0, np.float32), sr)) == 0)

loud = (0.1 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
l = measure_lufs(loud, sr)
print("LUFS of 0.1-amplitude 1 kHz sine:", l, "(expect about -23)")
check("LUFS plausible", l is not None and abs(l - (-23.0)) < 0.7)
check("LUFS short audio -> None", measure_lufs(loud[:1000], sr) is None)
check("LUFS silence -> None", measure_lufs(z, sr) is None)

print(f"\n{sum(ok)}/{len(ok)} checks passed")