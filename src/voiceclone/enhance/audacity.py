"""Remove analogue noise (hiss, mains hum, clicks) with Audacity 3.x via mod-script-pipe.

Audacity 4 has no scripting; use 3.x with mod-script-pipe enabled.
Per recording:
  1. decode to 44.1 kHz mono, untouched;
  2. find the quietest 1 s (noise profile) and exact mains-hum peaks near
     50/100/150/200 Hz. If the quietest second is below -78 dB the pauses were
     digitally gated during transfer: there is no hiss-only audio to profile,
     so Noise Reduction is skipped (a profile of breaths would strip voice);
  3. restart Audacity (Noise Reduction keeps one profile per session), then
     profile -> NoiseReduction x2 (~12 dB) -> NotchFilter x2 per hum (Q 20)
     -> ClickRemoval -> export;
  4. optionally re-cut chunks at an earlier segmentation's times, so its
     verbatim transcripts can be reused unchanged.
"""
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

import numpy as np

DEFAULT_APP = Path.home() / "Applications/Audacity 3.7.app"
GATED_DB = -78.0


class AudacityPipe:
    """Minimal client for mod-script-pipe named pipes (macOS/Linux)."""

    def __init__(self):
        uid = os.getuid()
        self.to_path = f"/tmp/audacity_script_pipe.to.{uid}"
        self.from_path = f"/tmp/audacity_script_pipe.from.{uid}"
        self._to = self._from = None

    def ready(self) -> bool:
        return all(os.path.exists(p) and stat.S_ISFIFO(os.stat(p).st_mode) for p in (self.to_path, self.from_path))

    def reset(self):
        for p in (self.to_path, self.from_path):
            if os.path.exists(p):
                os.remove(p)
        self._to = self._from = None

    def cmd(self, command: str) -> str:
        if self._to is None:
            if not self.ready():
                raise RuntimeError("Audacity script pipes are not up (Audacity 3.x with mod-script-pipe?)")
            self._to = os.fdopen(os.open(self.to_path, os.O_WRONLY), "w")   # never create a plain file
            self._from = open(self.from_path, "r")
        self._to.write(command + "\n"); self._to.flush()
        lines = []
        while True:
            line = self._from.readline()
            if line == "\n" and lines:
                break
            lines.append(line)
        out = "".join(lines)
        if "BatchCommand finished: Failed" in out:
            raise RuntimeError(f"Audacity command failed: {command}\n{out}")
        return out


def analyse_noise(y: np.ndarray, sr: int):
    """(start of quietest 1 s, its level in dB, hum frequencies with >10 dB prominence)."""
    hop = sr // 50; n = len(y) // hop
    db = 20 * np.log10(np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    sm = np.convolve(db, np.ones(50) / 50, "valid")
    seg = y[: min(len(y), 300 * sr)]                    # first 5 min is enough to locate hum
    F = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))); f = np.fft.rfftfreq(len(seg), 1 / sr)
    hums = []
    for h in (50, 100, 150, 200):
        m = (f > h - 3) & (f < h + 3)
        prom = 20 * np.log10(F[m].max() / np.median(F[(f > h - 15) & (f < h + 15)]))
        if prom > 10:
            hums.append(round(float(f[m][np.argmax(F[m])]), 2))
    return float(np.argmin(sm)) / 50, float(sm.min()), hums


def is_running(app: Path) -> bool:
    exe = f"{app}/Contents/MacOS/Audacity"
    return subprocess.run(["pgrep", "-f", exe], capture_output=True).returncode == 0


def _restart(app: Path, pipe: AudacityPipe):
    """Quit and relaunch Audacity: Noise Reduction keeps one profile per session."""
    subprocess.run(["pkill", "-f", f"{app}/Contents/MacOS/Audacity"])
    for _ in range(60):
        if not is_running(app):
            break
        time.sleep(0.5)
    pipe.reset()
    for _ in range(3):
        subprocess.run(["open", "-a", str(app)])
        for _ in range(60):
            if pipe.ready():
                break
            time.sleep(1)
        if pipe.ready():
            time.sleep(2)
            return
    raise RuntimeError("Audacity did not start with its script pipes after 3 attempts")


def audacity_denoise(src, work_dir, app=DEFAULT_APP, reuse_chunks=None, chunks_out=None,
                     force_restart: bool = False, log=print) -> Path:
    """Clean `src` into work_dir/audacity_44k.wav; optionally re-cut chunks (see module doc).

    Audacity is quit and relaunched for every file. If it is already open when
    this starts, it refuses to run unless `force_restart`, because quitting
    would discard unsaved work in the open project.
    """
    import librosa
    import soundfile as sf

    from voiceclone.audio import ffmpeg_decode, peak_normalize, write_wav

    if is_running(Path(app)) and not force_restart:
        raise RuntimeError("Audacity is already running. Save your work and quit it, or pass "
                           "--force-restart to let this step quit it (unsaved changes are lost).")

    work = Path(work_dir); work.mkdir(parents=True, exist_ok=True)
    raw, out = work / "raw_44k.wav", work / "audacity_44k.wav"
    ffmpeg_decode(src, raw, 44100)
    y, sr = sf.read(raw)
    t0, level, hums = analyse_noise(y, sr)
    gated = level < GATED_DB
    log(f"quietest 1 s at {t0:.2f}s ({level:.1f} dB) -> "
        f"{'gated pauses, Noise Reduction skipped' if gated else 'hiss profile used'}; hum at {hums}")

    pipe = AudacityPipe()
    _restart(Path(app), pipe)
    c = pipe.cmd
    c(f'Import2: Filename="{raw}"')
    if not gated:
        c(f"Select: Start={t0} End={t0 + 1.0} Mode=Set")
        c("NoiseReduction:")                            # first call captures the profile
        c("SelectAll:")
        c("NoiseReduction:"); c("NoiseReduction:")
    c("SelectAll:")
    for h in hums:
        c(f"NotchFilter: FREQUENCY={h} Q=20"); c(f"NotchFilter: FREQUENCY={h} Q=20")
    c("ClickRemoval: Threshold=200 Width=20")
    c(f'Export2: Filename="{out}" NumChannels=1')
    c("SelectAll:"); c("RemoveTracks:")
    raw.unlink()

    if reuse_chunks:
        srcdir, dst = Path(reuse_chunks), Path(chunks_out)
        dst.mkdir(parents=True, exist_ok=True)
        z, zsr = librosa.load(str(out), sr=24000, mono=True)
        from voiceclone.jsonl import read_jsonl

        segs = read_jsonl(srcdir / "segments.jsonl")
        for s in segs:
            x = z[int(max(0, s["start"] - 0.1) * zsr): int((s["end"] + 0.1) * zsr)]
            write_wav(dst / s["file"], peak_normalize(x), zsr)
        shutil.copy(srcdir / "segments.jsonl", dst); shutil.copy(srcdir / "transcripts.jsonl", dst)
        log(f"{len(segs)} chunks cut to {dst}")
    return out
