import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
from faster_whisper.audio import decode_audio
from scipy.signal import spectrogram

from config import SR
from denoise import spectral_subtraction
from editor import words_to_text
from pipeline import process
from transcribe import transcribe_words

DENOISERS = {"Spectral subtraction": "spectral", "None": "none"}


@st.cache_data(show_spinner="Denoising and transcribing (the first run loads the model)...")
def get_words(audio_bytes, denoiser, model_size):
    x = decode_audio(io.BytesIO(audio_bytes), sampling_rate=SR)
    x = np.asarray(x, dtype=np.float32)
    d = spectral_subtraction(x, SR) if denoiser == "spectral" else x
    return x, transcribe_words(d, SR, model_size=model_size)


def wav_bytes(y):
    buf = io.BytesIO()
    sf.write(buf, y, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def fmt(v, unit=""):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.1f}{unit}"


def waveform_fig(stages, final):
    items = [("Input", stages["input"]), ("Denoised", stages["denoised"]), ("Final", final)]
    fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    for ax, (title, y) in zip(axes, items):
        ax.plot(np.arange(len(y)) / SR, y, linewidth=0.5)
        ax.set_ylim(-1, 1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("time (s)")
    fig.tight_layout()
    return fig


def spectrogram_fig(stages, final):
    items = [("Input", stages["input"]), ("Denoised", stages["denoised"]), ("Final", final)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (title, y) in zip(axes, items):
        f, t, S = spectrogram(y, fs=SR, nperseg=512, noverlap=384)
        db = 10 * np.log10(S + 1e-12)
        ax.pcolormesh(t, f, db, shading="auto", cmap="magma",
                      vmin=db.max() - 70, vmax=db.max())
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Hz")
    fig.tight_layout()
    return fig


st.set_page_config(page_title="Voice Cleanup Studio", layout="wide")
st.title("Voice Cleanup Studio")
st.caption("Denoise, remove fillers, edit by transcript, trim silences, normalize loudness.")

with st.sidebar:
    st.header("Settings")
    denoiser_name = st.selectbox("Denoiser", list(DENOISERS))
    denoiser = DENOISERS[denoiser_name]
    size = st.selectbox("Whisper model", ["base", "small"])
    drop_fillers = st.checkbox("Remove fillers Whisper kept", value=True)
    snap = st.checkbox("Snap cuts inward to quiet points", value=True)
    trim = st.checkbox("Trim long silences", value=True)
    normalize = st.checkbox("Normalize loudness", value=True)
    target = st.slider("Target RMS (dBFS)", -30, -10, -20)
    st.caption("The denoiser learns the noise from the first 0.5 s, "
               "so start recordings with half a second of background noise only.")
    st.caption("After changing options, press Proceed again.")

up = st.file_uploader("Audio file", type=["wav", "flac", "mp3", "m4a"])
if not up:
    st.info("Upload a voice recording to begin.")
    st.stop()

data = up.getvalue()
x, words = get_words(data, denoiser, size)
sig = (up.name, denoiser, size)

st.subheader("1. Original")
st.audio(data)
st.caption(f"{len(x) / SR:.1f} s, {len(words)} words transcribed")

st.subheader("2. Edit the transcript")
st.caption("Delete words you want cut from the audio. Only deletions create cuts.")
key = f"text_{up.name}_{denoiser}_{size}"
if key not in st.session_state:
    st.session_state[key] = words_to_text(words)
edited = st.text_area("Transcript", key=key, height=200)

if st.button("Proceed", type="primary"):
    with st.spinner("Processing..."):
        res = process(x, SR, denoiser=denoiser, words=words, edited_text=edited,
                      remove_fillers=drop_fillers, snap=snap, trim=trim,
                      normalize=normalize, target_dbfs=target)
    st.session_state["result"] = res
    st.session_state["result_sig"] = sig

res = st.session_state.get("result") if st.session_state.get("result_sig") == sig else None

if res:
    s = res["stats"]
    final = res["audio"]
    st.subheader("3. Result")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", f"{s['duration_out']:.1f} s", f"{s['duration_out'] - s['duration_in']:.1f} s")
    c2.metric("Cuts", s["n_cuts"], f"{s['seconds_cut']:.2f} s removed", delta_color="off")
    c3.metric("RMS (dBFS)", fmt(s["rms_dbfs_out"]), f"was {fmt(s['rms_dbfs_in'])}", delta_color="off")
    c4.metric("LUFS", fmt(s["lufs_out"]), f"was {fmt(s['lufs_in'])}", delta_color="off")

    a, b, c = st.columns(3)
    a.caption("Denoised only")
    a.audio(wav_bytes(res["stages"]["denoised"]), format="audio/wav")
    b.caption("After cuts and trimming")
    b.audio(wav_bytes(res["stages"]["trimmed"]), format="audio/wav")
    c.caption("Final")
    c.audio(wav_bytes(final), format="audio/wav")

    st.download_button("Download final WAV", wav_bytes(final),
                       file_name="cleaned.wav", mime="audio/wav")

    if res["cuts"]:
        st.write("Cut ranges (in the original timeline):")
        st.dataframe([{"start (s)": round(s_, 2), "end (s)": round(e_, 2),
                       "length (s)": round(e_ - s_, 2)} for s_, e_ in res["cuts"]],
                     use_container_width=True)
    else:
        st.write("No cuts were made.")

    tab1, tab2 = st.tabs(["Waveforms", "Spectrograms"])
    with tab1:
        fig = waveform_fig(res["stages"], final)
        st.pyplot(fig)
        plt.close(fig)
    with tab2:
        fig = spectrogram_fig(res["stages"], final)
        st.pyplot(fig)
        plt.close(fig)