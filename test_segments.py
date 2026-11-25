import glob
import numpy as np
import soundfile as sf
from segments import remove_segments, snap_to_quiet, snap_ranges
from transcribe import transcribe_words
from editor import words_to_text, deleted_ranges

SR = 16000
tone = (0.5 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)).astype(np.float32)
before = tone.copy()
results = []


def check(name, ok, info=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"  -> {info}"))
    results.append(ok)


# ---- Part 1: synthetic checks ----
same = remove_segments(tone, SR, [])
check("no ranges: same audio, new array", np.array_equal(same, tone) and same is not tone)

mid = remove_segments(tone, SR, [(0.4, 0.6)], fade_ms=15)
check("middle cut length", len(mid) == 12560, f"got {len(mid)}")
check("fade_ms=0 length", len(remove_segments(tone, SR, [(0.4, 0.6)], fade_ms=0)) == 12800)
check("float32 output, input untouched", mid.dtype == np.float32 and np.array_equal(tone, before))

cut = [(0.40057, 0.6)]  # chosen so a plain cut makes a big jump
hard = np.abs(np.diff(remove_segments(tone, SR, cut, fade_ms=0))).max()
soft = np.abs(np.diff(remove_segments(tone, SR, cut, fade_ms=15))).max()
print(f"      biggest sample jump: hard cut {hard:.3f}, crossfade {soft:.3f}")
check("crossfade removes the click", soft < 0.2 and soft < hard / 2)

start = remove_segments(tone, SR, [(0, 0.5)])
check("cut at start", len(start) == 8000 and abs(start[0]) < 1e-6)
end = remove_segments(tone, SR, [(0.5, 1.0)])
check("cut at end", len(end) == 8000 and abs(end[-1]) < 1e-6)
check("overlapping ranges merge",
      np.array_equal(remove_segments(tone, SR, [(0.2, 0.5), (0.4, 0.6)]),
                     remove_segments(tone, SR, [(0.2, 0.6)])))
check("ranges outside the audio", len(remove_segments(tone, SR, [(-1, 0.1), (0.9, 5)])) == 12800)
check("cut everything", len(remove_segments(tone, SR, [(0, 1)])) == 0)
check("empty input", len(remove_segments(np.zeros(0, np.float32), SR, [(0, 1)])) == 0)

x = tone.copy()
x[4800:5600] = 0  # silence from 0.30 to 0.35 s
s1 = snap_to_quiet(x, SR, 0.29)
check("snap moves into the silence", 0.299 <= s1 <= 0.35, f"got {s1:.4f}")
s2 = snap_to_quiet(x, SR, 0.325)
check("snap stays put when already quiet", abs(s2 - 0.325) < 0.002, f"got {s2:.4f}")
check("snap clamps past the end", snap_to_quiet(x, SR, 5.0) <= 1.0)

check("side='after' never moves earlier", snap_to_quiet(x, SR, 0.29, 50, side="after") >= 0.29)
check("side='before' never moves later", snap_to_quiet(x, SR, 0.34, 50, side="before") <= 0.34)
print("\nall checks passed\n" if all(results) else "\nSOME CHECKS FAILED\n")

# ---- Part 2: real clip, delete "conductor", hard cut vs snapped + crossfaded ----
files = sorted(glob.glob("test_data/speech/*.flac"))
clip, sr = sf.read(files[0], dtype="float32")
words = transcribe_words(clip, sr)
ranges = deleted_ranges(words, words_to_text(words).replace("conductor ", ""))
# snapped = [(snap_to_quiet(clip, sr, s), snap_to_quiet(clip, sr, e)) for s, e in ranges]
snapped = snap_ranges(clip, sr, ranges)
print("cut ranges:", [(round(s, 3), round(e, 3)) for s, e in ranges])
print("snapped   :", [(round(s, 3), round(e, 3)) for s, e in snapped])
sf.write("test_data/out/cut_hard.wav", remove_segments(clip, sr, ranges, fade_ms=0), sr)
sf.write("test_data/out/cut_faded.wav", remove_segments(clip, sr, snapped, fade_ms=15), sr)
print("wrote test_data/out/cut_hard.wav and test_data/out/cut_faded.wav")