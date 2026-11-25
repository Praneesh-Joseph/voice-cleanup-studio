import re
from difflib import SequenceMatcher


def norm_token(token):
    """Lowercase and strip punctuation so 'Hello,' and 'hello' count as the same word."""
    return re.sub(r"[^\w']+", "", token.lower())


def words_to_text(words):
    return " ".join(w["word"] for w in words)


def merge_ranges(ranges, gap=0.0):
    """Sort and merge overlapping (or within `gap` seconds) (start, end) ranges."""
    out = []
    for s, e in sorted(ranges):
        if e <= s:
            continue
        if out and s <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def deleted_ranges(words, edited_text):
    """Compare the original words with the user's edited text and return the
    time ranges of the deleted words as sorted, non-overlapping (start, end) tuples.
    Only deletions cut audio. Retyped words are treated as spelling fixes."""
    orig_idx = [i for i, w in enumerate(words) if norm_token(w["word"])]
    orig = [norm_token(words[i]["word"]) for i in orig_idx]
    edited = [t for t in (norm_token(t) for t in edited_text.split()) if t]

    sm = SequenceMatcher(None, orig, edited, autojunk=False)
    ranges = []
    for tag, i1, i2, _, _ in sm.get_opcodes():
        if tag == "delete":
            first, last = words[orig_idx[i1]], words[orig_idx[i2 - 1]]
            ranges.append((first["start"], last["end"]))  # one range per run of deleted words
    return merge_ranges(ranges)