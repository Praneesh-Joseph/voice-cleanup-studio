import tarfile
import urllib.request
from pathlib import Path

MIRRORS = [
    "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "https://us.openslr.org/resources/12/dev-clean.tar.gz",
    "https://eu.openslr.org/resources/12/dev-clean.tar.gz",
]
OUT = Path("test_data") / "speech"
N_SPEAKERS = 5
PER_SPEAKER = 2


def fetch(url):
    OUT.mkdir(parents=True, exist_ok=True)
    per_speaker = {}
    saved = 0
    with urllib.request.urlopen(url, timeout=30) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if not member.name.endswith(".flac"):
                    continue
                speaker = member.name.split("/")[2]
                if per_speaker.get(speaker, 0) >= PER_SPEAKER:
                    continue
                if speaker not in per_speaker and len(per_speaker) >= N_SPEAKERS:
                    continue
                data = tar.extractfile(member).read()
                (OUT / Path(member.name).name).write_bytes(data)
                per_speaker[speaker] = per_speaker.get(speaker, 0) + 1
                saved += 1
                print(f"saved {member.name.split('/')[-1]}  ({saved}/{N_SPEAKERS * PER_SPEAKER})")
                if saved == N_SPEAKERS * PER_SPEAKER:
                    break
    return saved


if __name__ == "__main__":
    existing = list(OUT.glob("*.flac")) if OUT.exists() else []
    if len(existing) >= N_SPEAKERS * PER_SPEAKER:
        print("Files already downloaded, nothing to do.")
    else:
        for url in MIRRORS:
            try:
                print("Trying", url)
                if fetch(url) > 0:
                    break
            except Exception as e:
                print("Failed:", e)
        else:
            print("All mirrors failed. Tell me the error and we'll find another way.")

    # quick check that the files are readable
    import soundfile as sf
    for f in sorted(OUT.glob("*.flac")):
        info = sf.info(str(f))
        print(f.name, info.samplerate, "Hz,", info.channels, "ch,", round(info.duration, 1), "s")