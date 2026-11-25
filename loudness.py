import numpy as np


def rms_normalize(x, sr, target_dbfs=-20.0, peak_limit=0.97):
    """Scale x so its RMS is target_dbfs. If that would push the peak above
    peak_limit, scale down so the peak sits at peak_limit instead.
    Returns a new float32 array (same length); silence is returned unchanged."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x.copy()
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    if rms < 1e-8:
        return x.copy()
    gain = (10.0 ** (target_dbfs / 20.0)) / rms
    y = x.astype(np.float64) * gain
    peak = float(np.max(np.abs(y)))
    if peak > peak_limit:
        y *= peak_limit / peak
    return y.astype(np.float32)


def rms_dbfs(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("-inf")
    rms = np.sqrt(np.mean(x ** 2))
    return float(20 * np.log10(rms)) if rms > 1e-12 else float("-inf")


def measure_lufs(x, sr):
    """Integrated loudness in LUFS (ITU-R BS.1770) via pyloudnorm.
    Reported only, never used to change the audio.
    Returns None if the audio is too short (< 0.4 s) or silent."""
    import pyloudnorm as pyln
    x = np.asarray(x, dtype=np.float64)
    if x.size < int(0.4 * sr):
        return None
    try:
        val = pyln.Meter(sr).integrated_loudness(x)
    except ValueError:
        return None
    return float(val) if np.isfinite(val) else None