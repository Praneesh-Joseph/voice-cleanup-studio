import numpy as np
from config import SR
from denoise import spectral_subtraction, wiener_filter
from segments import remove_segments, snap_ranges
from vad import trim_silences
from loudness import rms_normalize, measure_lufs, rms_dbfs


def _merge(ranges):
    """Sort and merge overlapping (start, end) ranges."""
    out = []
    for s, e in sorted((float(s), float(e)) for s, e in ranges if e > s):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def process(x, sr=SR, denoiser="spectral", words=None, edited_text=None,
            remove_fillers=True, snap=True, trim=True, normalize=True,
            target_dbfs=-20.0, model_size="base", progress=None):
    """Full Voice Cleanup Studio pipeline.

    denoiser: "spectral" | "wiener" | "none"
    words: previously transcribed words (skips Whisper if given)
    edited_text: transcript text after the user's edits (None = no manual cuts)
    Returns a dict: audio, words, cuts, stages, stats.
    """
    def say(msg):
        if progress:
            progress(msg)

    x = np.asarray(x, dtype=np.float32)
    stages = {"input": x}

    # 1. denoise (output has the same length, so timestamps stay valid)
    say("Denoising...")
    if denoiser == "spectral":
        y = spectral_subtraction(x, sr)
    elif denoiser == "wiener":
        y = wiener_filter(x, sr)
    else:
        y = x.copy()
    stages["denoised"] = y

    # 2. Whisper word timestamps on the denoised audio (only if needed)
    need_words = remove_fillers or edited_text is not None
    if need_words and words is None:
        say("Transcribing (first time can take a while)...")
        from transcribe import transcribe_words
        words = transcribe_words(y, sr, model_size=model_size)
    words = words or []

    # 3. one merged cut list: fillers + manual transcript deletions
    cuts = []
    if remove_fillers and words:
        from transcribe import find_fillers
        cuts += list(find_fillers(words))
    if edited_text is not None and words:
        from editor import deleted_ranges
        cuts += list(deleted_ranges(words, edited_text))
    cuts = _merge(cuts)
    if snap and cuts:
        cuts = _merge(snap_ranges(y, sr, cuts))   # inward-only snapping

    # 4. remove_segments, called once
    if cuts:
        say("Applying cuts...")
        y = remove_segments(y, sr, cuts, fade_ms=15)
    stages["cut"] = y

    # 5. silence trimming
    if trim:
        say("Trimming silences...")
        r = trim_silences(y, sr, min_silence=0.7, keep_silence=0.3)
        y = r[0] if isinstance(r, tuple) else r
    stages["trimmed"] = y

    # 6. loudness
    if normalize:
        say("Normalizing loudness...")
        y = rms_normalize(y, sr, target_dbfs=target_dbfs)
    stages["final"] = y

    stats = {
        "duration_in": len(x) / sr,
        "duration_out": len(y) / sr,
        "n_cuts": len(cuts),
        "seconds_cut": float(sum(e - s for s, e in cuts)),
        "rms_dbfs_in": rms_dbfs(x),
        "rms_dbfs_out": rms_dbfs(y),
        "lufs_in": measure_lufs(x, sr),
        "lufs_out": measure_lufs(y, sr),
    }
    say("Done.")
    return {"audio": y, "words": words, "cuts": cuts,
            "stages": stages, "stats": stats}