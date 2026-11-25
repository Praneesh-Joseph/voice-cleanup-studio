import glob
import soundfile as sf
from config import SR
from transcribe import transcribe_words
from editor import words_to_text

files = sorted(glob.glob("test_data/speech/*.flac"))
x, sr = sf.read(files[0], dtype="float32")
assert sr == SR and x.ndim == 1

words = transcribe_words(x, sr)
print(files[0], f"({len(x)/sr:.1f} s)")
for w in words:
    print(f'{w["start"]:6.2f} -> {w["end"]:6.2f}  {w["word"]}')
print()
print(words_to_text(words))