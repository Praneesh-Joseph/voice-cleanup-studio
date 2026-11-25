import io
import numpy as np
import soundfile as sf
import streamlit as st
from faster_whisper.audio import decode_audio
from transcribe import transcribe_words, find_fillers
from editor import words_to_text, deleted_ranges, merge_ranges

SR = 16000


@st.cache_data(show_spinner="Transcribing...")
def get_words(audio_bytes, model_size):
    x = decode_audio(io.BytesIO(audio_bytes), sampling_rate=SR)
    return x, transcribe_words(x, SR, model_size=model_size)


st.title("Transcript editor demo")
up = st.file_uploader("Audio file", type=["wav", "flac", "mp3", "m4a"])
size = st.selectbox("Whisper model", ["base", "small"])

if up:
    data = up.getvalue()
    x, words = get_words(data, size)
    st.audio(data)

    key = f"text_{up.name}_{size}"
    if key not in st.session_state:
        st.session_state[key] = words_to_text(words)
    st.caption("Delete words to cut them from the audio.")
    edited = st.text_area("Transcript", key=key, height=200)
    drop_fillers = st.checkbox("Also remove fillers Whisper kept", value=True)

    if st.button("Proceed"):
        cuts = deleted_ranges(words, edited)
        if drop_fillers:
            cuts = merge_ranges(cuts + find_fillers(words))
        st.write("Cut ranges:", [(round(s, 2), round(e, 2)) for s, e in cuts])

        pieces, prev = [], 0
        for s, e in cuts:
            pieces.append(x[prev:int(s * SR)])
            prev = int(e * SR)
        pieces.append(x[prev:])

        buf = io.BytesIO()
        sf.write(buf, np.concatenate(pieces), SR, format="WAV")
        st.audio(buf.getvalue(), format="audio/wav")