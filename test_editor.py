import glob
import numpy as np
import soundfile as sf
from editor import words_to_text, deleted_ranges
from transcribe import transcribe_words, find_fillers

# ---- Part 1: fake words, no Whisper needed ----
words = [
    {"word": "Hello,", "start": 0.0, "end": 0.5},
    {"word": "um,",    "start": 0.6, "end": 0.9},
    {"word": "my",     "start": 1.0, "end": 1.2},
    {"word": "name",   "start": 1.2, "end": 1.5},
    {"word": "is",     "start": 1.5, "end": 1.7},
    {"word": "uh,",    "start": 1.9, "end": 2.2},
    {"word": "Sam.",   "start": 2.3, "end": 2.8},
]


def check(name, got, want):
    ok = got == want
    print(("PASS  " if ok else "FAIL  ") + name, "" if ok else f"-> got {got}, want {want}")
    return ok


results = [
    check("unchanged text", deleted_ranges(words, words_to_text(words)), []),
    check("delete first um", deleted_ranges(words, "Hello, my name is, uh, Sam."), [(0.6, 0.9)]),
    check("delete both fillers", deleted_ranges(words, "Hello, my name is Sam."), [(0.6, 0.9), (1.9, 2.2)]),
    check("delete two words in a row", deleted_ranges(words, "Hello, um, is, uh, Sam."), [(1.0, 1.5)]),
    check("case/punctuation only", deleted_ranges(words, "hello um my name is uh sam"), []),
    check("retyped word is not a cut", deleted_ranges(words, "Hello, um, my name is, uh, Samuel."), []),
    check("delete everything", deleted_ranges(words, ""), [(0.0, 2.8)]),
    check("find_fillers", find_fillers(words), [(0.6, 0.9), (1.9, 2.2)]),
]
print("\nall checks passed\n" if all(results) else "\nSOME CHECKS FAILED\n")

# ---- Part 2: real clip, delete the word "conductor" and cut the audio ----
files = sorted(glob.glob("test_data/speech/*.flac"))
x, sr = sf.read(files[0], dtype="float32")
real = transcribe_words(x, sr)
text = words_to_text(real).replace("conductor ", "")
ranges = deleted_ranges(real, text)
print("edited text:", text)
print("cut ranges :", ranges)

pieces, prev = [], 0
for s, e in ranges:
    pieces.append(x[prev:int(s * sr)])
    prev = int(e * sr)
pieces.append(x[prev:])
sf.write("test_data/out/editor_cut.wav", np.concatenate(pieces), sr)
print("wrote test_data/out/editor_cut.wav")