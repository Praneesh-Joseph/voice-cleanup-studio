import sys
from faster_whisper.audio import decode_audio
from transcribe import transcribe_words, DISFLUENCY_PROMPT

FILLERS = {"um", "umm", "uh", "uhh", "er", "erm", "ah", "hmm", "hm", "mm", "mhm"}

path = sys.argv[1]
x = decode_audio(path, sampling_rate=16000)  # mono float32 at 16 kHz, handles m4a/mp3/wav
print(f"{path}: {len(x)/16000:.1f} s\n")


def run(label, size, prompt):
    words = transcribe_words(x, 16000, model_size=size, prompt=prompt)
    marked, n_fill = [], 0
    for w in words:
        if w["word"].lower().strip(".,!?…-") in FILLERS:
            marked.append(f'[{w["word"]}]')
            n_fill += 1
        else:
            marked.append(w["word"])
    print(f"=== {label} ===")
    print(" ".join(marked))
    print(f"filler words found: {n_fill}")
    gaps = [(a["end"], b["start"]) for a, b in zip(words, words[1:]) if b["start"] - a["end"] > 0.2]
    print("gaps > 200 ms:", ", ".join(f"{s:.2f}-{e:.2f}" for s, e in gaps) or "none")
    print()


run("base + prompt", "base", DISFLUENCY_PROMPT)
run("small + prompt", "small", DISFLUENCY_PROMPT)
run("small, no prompt", "small", None)