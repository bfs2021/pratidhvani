"""Trim synthesized samples right after the last word of their target sentence.

A fine-tuned model sometimes keeps talking past the sentence. The target text
is force-aligned to the generated audio with a trailing <star> that soaks up
any extra babble, and the audio is cut `tail` seconds after the last word.
"""
from pathlib import Path

from voiceclone.audio import write_wav
from voiceclone.evaluate import sample_texts
from voiceclone.text import star_tokens


def trim_to_text(aligner, sample_dir, sentences: list[str] | None = None, out_dir=None, tail: float = 0.3, log=print) -> Path:
    import librosa
    import soundfile as sf

    src = Path(sample_dir)
    dst = Path(out_dir) if out_dir else src.parent / (src.name + "_trimmed")
    dst.mkdir(parents=True, exist_ok=True)
    for name, text in sample_texts(src, sentences).items():
        wav, sr = sf.read(src / name, dtype="float32")
        toks, txt = star_tokens(text)
        end = len(wav) / sr
        if toks:
            em, stride = aligner.emissions(librosa.resample(wav, orig_sr=sr, target_sr=16000))
            words = aligner.align_tokens(em, stride, toks, txt)
            end = min(end, words[-1]["end"] + tail)
        write_wav(dst / name, wav[: int(end * sr)], sr)
        log(f"{name}: {len(wav)/sr:.1f}s -> {end:.1f}s | {text[:50]}")
    if (src / "log.json").exists():          # keep each sample's text with the trimmed copy
        import shutil

        shutil.copy(src / "log.json", dst / "log.json")
    return dst
