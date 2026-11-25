# def mix_at_snr(clean, noise, snr_db):
#     """STUB: will mix noise into clean speech at the given SNR (dB)."""
#     return clean.copy()


# def snr_db(clean, test):
#     """STUB: will return SNR of `test` against `clean` in dB."""
#     return 0.0


import numpy as np


def mix_at_snr(clean, noise, snr_db):
    """Add noise to clean speech so the result has exactly `snr_db` dB SNR."""
    clean = np.asarray(clean, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)
    if len(noise) < len(clean):  # loop the noise if it's too short
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    noise = noise[:len(clean)]

    p_clean = np.mean(clean ** 2)
    p_noise = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(p_clean / (p_noise * 10 ** (snr_db / 10)))
    return (clean + scale * noise).astype(np.float32)


def snr_db(clean, test):
    """SNR of `test` against the clean reference, in dB (higher = closer to clean)."""
    clean = np.asarray(clean, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    n = min(len(clean), len(test))
    clean, test = clean[:n], test[:n]
    err = clean - test
    return 10 * np.log10((np.sum(clean ** 2) + 1e-12) / (np.sum(err ** 2) + 1e-12))


def make_test_pair(clean, noise, target_db, sr, lead_sec=0.5):
    """Prepend `lead_sec` of silence to the clean speech, then mix in noise.

    The silence becomes a noise-only lead-in, which is what the denoiser uses
    as its noise fingerprint. Returns (clean_padded, noisy).
    """
    lead = np.zeros(int(lead_sec * sr), dtype=np.float32)
    clean_p = np.concatenate([lead, np.asarray(clean, dtype=np.float32)])
    return clean_p, mix_at_snr(clean_p, noise, target_db)