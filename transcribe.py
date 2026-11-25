import numpy as np
from faster_whisper import WhisperModel
from editor import merge_ranges, norm_token

_models = {}

# A prompt full of disfluencies nudges Whisper to keep "um"/"uh" instead of cleaning them up
DISFLUENCY_PROMPT = "Umm, let me think, like, hmm... Okay, so, uh, here's what I'm, like, thinking."


def _get_model(model_size):
    if model_size not in _models:
        _models[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _models[model_size]


def transcribe_words(x, sr, model_size="base", prompt=DISFLUENCY_PROMPT):
    """x: 1-D float32 mono array at 16 kHz. Returns [{"word", "start", "end"}, ...]."""
    if sr != 16000:
        raise ValueError("transcribe_words expects sr=16000")
    model = _get_model(model_size)
    segments, _ = model.transcribe(
        x.astype(np.float32),
        language="en",
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=False,
        initial_prompt=prompt,
    )
    words = []
    for seg in segments:
        for w in seg.words:
            words.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end)})
    return words


# def find_fillers(words):
#     """Stub: returns sorted non-overlapping (start_sec, end_sec) ranges."""
#     return []

FILLER_WORDS = {"um", "umm", "uh", "uhh", "er", "erm", "ah", "hmm", "hm", "mm", "mhm"}


def find_fillers(words, fillers=FILLER_WORDS):
    """Words Whisper kept that are fillers -> sorted non-overlapping (start_sec, end_sec) ranges."""
    ranges = [(w["start"], w["end"]) for w in words if norm_token(w["word"]) in fillers]
    return merge_ranges(ranges)


# def find_gap_fillers(x, sr, words):
#     """Stub: returns sorted non-overlapping (start_sec, end_sec) ranges."""
#     return []




def _frame_db(x, sr, frame_ms=25, hop_ms=10):
    n = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if len(x) < n:
        return np.array([]), hop / sr
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    db = 10 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    return db, hop / sr

def gap_candidates(x, sr, words, min_gap=0.2, min_len=0.12, max_len=1.2, margin=0.03):
    """Energy bursts inside gaps between Whisper words = likely um/uh.
    Returns sorted non-overlapping (start_sec, end_sec) ranges."""
    db, hop_s = _frame_db(x, sr)
    if len(db) == 0 or len(words) < 2:
        return []
    lo, hi = np.percentile(db, 10), np.percentile(db, 95)   # noise floor vs. speech level
    active = db > lo + 0.35 * (hi - lo)
    ranges = []
    for a, b in zip(words, words[1:]):
        g0, g1 = a["end"] + margin, b["start"] - margin
        if g1 - g0 < min_gap:
            continue
        i, i1 = int(g0 / hop_s), min(int(g1 / hop_s), len(active))
        while i < i1:
            if active[i]:
                j = i
                while j < i1 and active[j]:
                    j += 1
                if min_len <= (j - i) * hop_s <= max_len:
                    ranges.append((i * hop_s, j * hop_s))
                i = j
            else:
                i += 1
    return ranges

def _voiced_fraction(seg, sr, fmin=70, fmax=400, thresh=0.45, frame_ms=40, hop_ms=10):
    """Fraction of frames with a clear periodic (pitched) component."""
    n = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if len(seg) < n:
        return 0.0
    lag_min, lag_max = int(sr / fmax), int(sr / fmin)
    voiced = total = 0
    for i in range(0, len(seg) - n + 1, hop):
        f = seg[i:i + n].astype(np.float64)
        f = f - f.mean()
        e = np.dot(f, f)
        total += 1
        if e < 1e-10:
            continue
        ac = np.correlate(f, f, mode="full")[n - 1:]
        if ac[lag_min:lag_max + 1].max() / ac[0] > thresh:
            voiced += 1
    return voiced / total if total else 0.0


def find_gap_fillers(x, sr, words, min_voiced=0.6, **kw):
    """Energy bursts in gaps between words that are also pitched = likely um/uh."""
    out = []
    for s, e in gap_candidates(x, sr, words, **kw):
        if _voiced_fraction(x[int(s * sr):int(e * sr)], sr) >= min_voiced:
            out.append((s, e))
    return out