import numpy as np


def _keep_intervals(n, sr, ranges):
    """Turn cut ranges (seconds) into the sample intervals to KEEP. Clips,
    sorts and merges overlapping ranges first."""
    cuts = []
    for s, e in sorted(ranges):
        a = max(0, int(round(s * sr)))
        b = min(n, int(round(e * sr)))
        if b <= a:
            continue
        if cuts and a <= cuts[-1][1]:
            cuts[-1] = (cuts[-1][0], max(cuts[-1][1], b))
        else:
            cuts.append((a, b))
    keep, prev = [], 0
    for a, b in cuts:
        if a > prev:
            keep.append((prev, a))
        prev = b
    if prev < n:
        keep.append((prev, n))
    return keep


def remove_segments(x, sr, ranges, fade_ms=15):
    """Remove the (start_sec, end_sec) ranges from x and crossfade every join.
    Never changes x. Returns a new float32 array. Each join is an overlap, so
    the output is fade_ms shorter than the plain cut would be."""
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    keep = _keep_intervals(n, sr, ranges)
    if not keep:
        return np.zeros(0, dtype=np.float32)
    if keep == [(0, n)]:
        return x.copy()

    fade = int(sr * fade_ms / 1000)
    out = x[keep[0][0]:keep[0][1]].copy()
    for a, b in keep[1:]:
        piece = x[a:b]
        k = min(fade, len(out), len(piece))
        if k > 0:
            t = np.linspace(0, np.pi / 2, k, dtype=np.float32)  # equal-power crossfade
            out[-k:] = out[-k:] * np.cos(t) + piece[:k] * np.sin(t)
        out = np.concatenate([out, piece[k:]])

    # If a cut touches the very start or end, fade there too so there is no click
    k = min(fade, len(out))
    if k > 0 and keep[0][0] > 0:
        out[:k] *= np.linspace(0, 1, k, dtype=np.float32)
    if k > 0 and keep[-1][1] < n:
        out[-k:] *= np.linspace(1, 0, k, dtype=np.float32)
    return out


# def snap_to_quiet(x, sr, t, window_ms=50):
#     """Move a cut time t (seconds) to the quietest nearby point (within +-window_ms),
#     then to the nearest zero crossing. Returns the new time in seconds."""
#     n = len(x)
#     t = float(min(max(t, 0.0), n / sr))
#     c = int(round(t * sr))
#     w = int(sr * window_ms / 1000)
#     lo, hi = max(0, c - w), min(n, c + w)
#     if hi - lo < 2:
#         return t
#     seg = np.abs(np.asarray(x[lo:hi], dtype=np.float64))
#     k = max(1, int(sr * 0.005))  # 5 ms smoothing
#     sm = np.convolve(seg, np.ones(k) / k, mode="same")
#     # tiny penalty so that ties (e.g. digital silence) pick the point closest to t
#     sm = sm + 1e-6 * np.abs(np.arange(len(sm)) - (c - lo)) / w
#     pos = lo + int(np.argmin(sm))

#     z = int(sr * 0.002)  # look for a zero crossing within 2 ms
#     a, b = max(1, pos - z), min(n - 1, pos + z)
#     if b > a:
#         s = np.signbit(x[a - 1:b])
#         zc = np.nonzero(s[1:] != s[:-1])[0]
#         if zc.size:
#             cand = a + zc
#             pos = int(cand[np.argmin(np.abs(cand - pos))])
#     return pos / sr



def snap_to_quiet(x, sr, t, window_ms=50, side="both"):
    """Move time t (seconds) to the quietest nearby point, then to the nearest zero crossing.
    side="both": search t +- window; "after": only t..t+window; "before": only t-window..t."""
    n = len(x)
    t = float(min(max(t, 0.0), n / sr))
    c = int(round(t * sr))
    w = int(sr * window_ms / 1000)
    lo = max(0, c - (0 if side == "after" else w))
    hi = min(n, c + (0 if side == "before" else w))
    if hi - lo < 2:
        return t
    seg = np.abs(np.asarray(x[lo:hi], dtype=np.float64))
    k = max(1, int(sr * 0.005))  # 5 ms smoothing
    sm = np.convolve(seg, np.ones(k) / k, mode="same")
    # tiny penalty so that ties (e.g. digital silence) pick the point closest to t
    sm = sm + 1e-6 * np.abs(np.arange(len(sm)) - (c - lo)) / max(w, 1)
    pos = lo + int(np.argmin(sm))

    z = int(sr * 0.002)  # look for a zero crossing within 2 ms
    a, b = max(1, pos - z), min(n - 1, pos + z)
    if b > a:
        s = np.signbit(x[a - 1:b])
        zc = np.nonzero(s[1:] != s[:-1])[0]
        if zc.size:
            cand = a + zc
            pos = int(cand[np.argmin(np.abs(cand - pos))])
    if side == "after":
        pos = max(pos, c)
    elif side == "before":
        pos = min(pos, c)
    return pos / sr


def snap_ranges(x, sr, ranges, window_ms=20):
    """Snap each cut inward (start later, end earlier) so it never eats a neighbouring word."""
    out = []
    for s, e in ranges:
        s2 = snap_to_quiet(x, sr, s, window_ms, side="after")
        e2 = snap_to_quiet(x, sr, e, window_ms, side="before")
        out.append((s2, e2) if e2 > s2 else (s, e))
    return out