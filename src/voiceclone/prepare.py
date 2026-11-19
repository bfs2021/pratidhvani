"""One-shot preparation of an extra recording for training:
decode -> light restore -> pause segmentation -> (optional) ClearerVoice denoise -> verbatim ASR.
"""
import shutil
import subprocess
from pathlib import Path

from voiceclone.audio import TTS_SR, ffmpeg_decode, load_wav, restore


def prepare_recording(src, tag: str, data_dir="data", lang: str = "kn", enhance_python: str | None = None,
                      enhance_wet: float = 1.0, log=print) -> Path:
    """Returns the folder with chunk WAVs, segments.jsonl and transcripts.jsonl.

    `enhance_python` is the interpreter of an environment with clearvoice
    installed (it needs numpy<2); without it the denoise step is skipped.
    """
    from voiceclone.asr import transcribe_dir
    from voiceclone.segment import vad_segment

    data = Path(data_dir)
    work = data / "work" / tag
    full48 = ffmpeg_decode(src, work / "full_48k.wav", 48000)
    clean = restore(full48, work / "full_24k_clean.wav")
    full48.unlink()
    vad = data / f"vad_{tag}"
    log(vad_segment(load_wav(clean, TTS_SR), TTS_SR, vad, prefix=tag))
    final = vad
    if enhance_python:
        final = data / f"vad_{tag}_se"
        subprocess.run([enhance_python, "-m", "voiceclone.enhance.clearvoice", str(vad), str(final),
                        "--steps", "se", "--wet", str(enhance_wet)], check=True)
        shutil.copy(vad / "segments.jsonl", final)
    transcribe_dir(final, lang=lang, log=log)
    log(f"done {tag}: {final}")
    return final
