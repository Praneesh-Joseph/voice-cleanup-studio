import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/1227121/files/{}_16k.zip?download=1"
NOISES = {"DKITCHEN": "kitchen", "STRAFFIC": "traffic", "PCAFETER": "cafeteria"}
OUT = Path("test_data") / "noise"
TMP = Path("test_data") / "_tmp.zip"


class RangeFile(io.RawIOBase):
    """File-like object that fetches only the bytes zipfile asks for."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            cr = r.headers.get("Content-Range")
            if r.status != 206 or not cr:
                raise OSError("server does not support partial downloads")
            self.size = int(cr.split("/")[-1])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        self.pos += len(data)
        return data


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for code, nice in NOISES.items():
        target = OUT / f"{nice}.wav"
        if target.exists():
            print(nice, "already there, skipping")
            continue
        url = URL.format(code)
        print("Getting", nice, "...")
        try:
            zf = zipfile.ZipFile(RangeFile(url))
        except Exception as e:
            print("  partial download not possible, fetching full zip (~110 MB):", e)
            urllib.request.urlretrieve(url, TMP)
            zf = zipfile.ZipFile(TMP)
        member = next(m for m in zf.namelist() if m.endswith("ch01.wav"))
        target.write_bytes(zf.read(member))
        zf.close()
        TMP.unlink(missing_ok=True)
        print("  saved", target)

    import soundfile as sf
    for f in sorted(OUT.glob("*.wav")):
        info = sf.info(str(f))
        print(f.name, info.samplerate, "Hz,", info.channels, "ch,", round(info.duration), "s")