import sys
import numpy as np
import soundfile as sf
from faster_whisper.audio import decode_audio
from transcribe import transcribe_words, find_gap_fillers

path = sys.argv[1]
x = decode_audio(path, sampling_rate=16000)
words = transcribe_words(x, 16000, model_size="small")
ranges = find_gap_fillers(x, 16000, words)

print(f"detected {len(ranges)} candidate fillers")
for s, e in ranges:
    print(f"{s:6.2f} -> {e:6.2f}  ({(e - s) * 1000:.0f} ms)")

pause = np.zeros(int(0.3 * 16000), dtype=np.float32)
pieces = []
for s, e in ranges:
    pieces += [x[int(s * 16000):int(e * 16000)], pause]
if pieces:
    sf.write("test_data/out/gap_fillers_only.wav", np.concatenate(pieces), 16000)
    print("wrote test_data/out/gap_fillers_only.wav")