import numpy as np
from segments import remove_segments


def frame_energy_db(x, sr, frame_ms=25, hop_ms=10):
    """Energy of each frame in dB (relative to full scale). Frame i covers samples
    [i*hop, i*hop + frame). Returns a 1-D array."""
    n = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    x = np.asarray(x, dtype=np.float64)
    if len(x) < n:
        return np.zeros(0)
    c = np.cumsum(np.r_[0.0, x * x])
    starts = np.arange(0, len(x) - n + 1, hop)
    return 10 * np.log10((c[starts + n] - c[starts]) / n + 1e-12)


def detect_speech(x, sr, frame_ms=25, hop_ms=10, level=0.35, min_speech=0.1,
                  merge_gap=0.25, pad=0.05, min_contrast_db=8.0):
    """Energy-based speech detection. Returns sorted non-overlapping (start_sec, end_sec).
    The threshold adapts to the recording: it sits `level` of the way from the noise floor
    (10th percentile) to the speech level (95th percentile)."""
    db = frame_energy_db(x, sr, frame_ms, hop_ms)
    if len(db) == 0:
        return []
    lo, hi = np.percentile(db, 10), np.percentile(db, 95)
    if hi - lo < min_contrast_db:
        return []  # flat signal: nothing stands out as speech
    active = db > lo + level * (hi - lo)

    hop, frame = hop_ms / 1000, frame_ms / 1000
    d = np.diff(np.r_[False, active, False].astype(np.int8))
    runs = [(i * hop, (j - 1) * hop + frame)
            for i, j in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0])]

    merged = []
    for s, e in runs:  # bridge short pauses inside speech
        if merged and s - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    dur = len(x) / sr
    out = []
    for s, e in merged:
        if e - s < min_speech:  # ignore clicks and blips
            continue
        s, e = max(0.0, s - pad), min(dur, e + pad)
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def silence_cuts(x, sr, min_silence=0.7, keep_silence=0.3, **kw):
    """Cut ranges that shorten every silence of at least min_silence seconds down to
    keep_silence seconds. If no speech is found, returns [] (leave the audio alone)."""
    speech = detect_speech(x, sr, **kw)
    if not speech:
        return []
    half, dur, cuts = keep_silence / 2, len(x) / sr, []
    if speech[0][0] >= min_silence:
        cuts.append((0.0, speech[0][0] - half))
    for (_, e1), (s2, _) in zip(speech, speech[1:]):
        if s2 - e1 >= min_silence:
            cuts.append((e1 + half, s2 - half))
    if dur - speech[-1][1] >= min_silence:
        cuts.append((speech[-1][1] + half, dur))
    return cuts


def trim_silences(x, sr, min_silence=0.7, keep_silence=0.3, **kw):
    """Return x with long silences shortened (crossfaded joins). Never changes x."""
    return remove_segments(x, sr, silence_cuts(x, sr, min_silence, keep_silence, **kw))