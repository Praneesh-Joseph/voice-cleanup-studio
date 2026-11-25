import soundfile as sf
from pathlib import Path
from evaluate import snr_db, make_test_pair

SR = 16000
speech_file = sorted(Path("test_data/speech").glob("*.flac"))[0]
clean, sr1 = sf.read(speech_file, dtype="float32")
noise, sr2 = sf.read("test_data/noise/kitchen.wav", dtype="float32")
assert sr1 == SR and sr2 == SR, "sample rate mismatch"

print("speech file:", speech_file.name, "| clean length:", len(clean))
for target in (0, 5, 10):
    clean_p, noisy = make_test_pair(clean, noise, target, SR)
    print(f"target {target:>2} dB -> measured {snr_db(clean_p, noisy):.3f} dB"
          f" | lengths {len(clean_p)} {len(noisy)}")