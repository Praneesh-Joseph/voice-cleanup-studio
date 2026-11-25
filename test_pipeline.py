import glob
import numpy as np
import soundfile as sf
from pipeline import process
from loudness import rms_dbfs

sr = 16000
rng = np.random.default_rng(0)
x = (0.05 * rng.standard_normal(sr * 6)).astype(np.float32)
ok = []

def check(name, cond):
    ok.append(cond); print(("PASS " if cond else "FAIL ") + name)

# fake words, so Whisper is not needed
words = [{"word": "hello", "start": 0.5, "end": 1.0},
         {"word": "um", "start": 2.0, "end": 2.4},
         {"word": "world", "start": 3.0, "end": 3.5}]

r = process(x, sr, denoiser="none", words=words, remove_fillers=True,
            snap=False, trim=False, normalize=False)
print("cuts:", r["cuts"], "duration:", r["stats"]["duration_out"])
check("filler cut found", len(r["cuts"]) == 1)
check("output shorter by about 0.4 s", 5.5 < r["stats"]["duration_out"] < 5.7)

r = process(x, sr, denoiser="none", words=words, remove_fillers=False,
            trim=False, normalize=False)
check("no cuts when fillers off", r["cuts"] == [] and len(r["audio"]) == len(x))

r = process(x, sr, denoiser="spectral", words=[], remove_fillers=False,
            trim=False, normalize=True)
check("normalized to -20 dBFS", abs(rms_dbfs(r["audio"]) + 20) < 0.1)
check("all stages present",
      set(r["stages"]) == {"input", "denoised", "cut", "trimmed", "final"})

# real clip with Whisper: delete "conductor"
f = glob.glob("test_data/**/1988-147956-0010*.flac", recursive=True)
if f:
    clip, _ = sf.read(f[0], dtype="float32")
    from editor import words_to_text
    r1 = process(clip, sr, denoiser="none", remove_fillers=False, trim=False, normalize=False,
                 edited_text=None)
    words = r1["words"] or __import__("transcribe").transcribe_words(clip, sr)
    text = words_to_text(words).replace("conductor", "").replace("  ", " ")
    r2 = process(clip, sr, denoiser="none", words=words, edited_text=text,
                 trim=False, normalize=False)
    print("cuts after deleting 'conductor':", r2["cuts"])
    sf.write("test_data/out/pipeline_demo.wav", r2["audio"], sr)
    check("one cut near 1.5-1.9 s", len(r2["cuts"]) == 1 and 1.4 < r2["cuts"][0][0] < 1.9)
else:
    print("LibriSpeech clip not found, skipped the Whisper test")

print(f"\n{sum(ok)}/{len(ok)} checks passed")