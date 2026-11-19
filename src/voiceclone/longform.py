"""Re-voice a stretch of a real recording with the cloned voice (same-words test).

Each verbatim chunk transcript up to `minutes` is synthesized and placed at
the original chunk's start time, or right after the previous clone chunk if
that ran long, so the two recordings stay roughly in step.

Writes cloned.wav, original.wav, ab.wav (his line, short gap, clone line,
longer gap) and script.tsv. Generated chunks are cached in chunks/ so an
interrupted run resumes.
"""
import time
from pathlib import Path

import numpy as np

from voiceclone.audio import TTS_SR, write_wav
from voiceclone.jsonl import read_jsonl
from voiceclone.svara.synth import GenParams
from voiceclone.text import collapse_spaces

LONGFORM_PARAMS = GenParams(min_sps=2.5)


def long_form(synth, chunks_dir, source_wav, ref_wav, ref_text: str, out_dir, minutes: float = 5.0,
              seed: int = 0, p: GenParams = LONGFORM_PARAMS, gap: float = 0.25, log=print) -> Path:
    import librosa
    import soundfile as sf

    segs = {s["file"]: s for s in read_jsonl(Path(chunks_dir) / "segments.jsonl")}
    rows = []
    for t in read_jsonl(Path(chunks_dir) / "transcripts.jsonl"):
        s = segs[t["file"]]
        if s["end"] <= minutes * 60:
            rows.append({"start": s["start"], "end": s["end"], "text": collapse_spaces(t["text"])})
    rows.sort(key=lambda r: r["start"])
    total = rows[-1]["end"]

    ref_tokens = synth.reference(ref_wav)
    out = Path(out_dir); (out / "chunks").mkdir(parents=True, exist_ok=True)
    placed, pieces, cursor = [], [], 0.0
    for i, r in enumerate(rows):
        if not r["text"]:
            pieces.append(None); continue
        cache = out / "chunks" / f"{i:03d}.wav"
        t0 = time.time()
        if cache.exists():
            wav = sf.read(cache, dtype="float32")[0]
        else:
            wav, _ = synth.generate(r["text"], ref_tokens, ref_text, seed=seed + i, p=p)
            write_wav(cache, wav, TTS_SR)
            log(f"[{i+1}/{len(rows)}] {r['start']:6.1f}s {len(wav)/TTS_SR:4.1f}s audio in {time.time()-t0:4.0f}s | {r['text'][:50]}")
        start = max(r["start"], cursor)
        placed.append((start, wav)); pieces.append(wav)
        cursor = start + len(wav) / TTS_SR + gap

    clone = np.zeros(int(max(cursor, total) * TTS_SR) + TTS_SR, dtype=np.float32)
    for start, wav in placed:
        a = int(start * TTS_SR); clone[a:a + len(wav)] = wav
    write_wav(out / "cloned.wav", clone, TTS_SR)
    orig, _ = librosa.load(str(source_wav), sr=TTS_SR, mono=True, duration=total)
    write_wav(out / "original.wav", orig, TTS_SR)
    gap_s, gap_l = np.zeros(int(0.4 * TTS_SR), np.float32), np.zeros(int(1.0 * TTS_SR), np.float32)
    ab = []
    for r, w in zip(rows, pieces):
        if w is not None:
            ab += [orig[int(r["start"] * TTS_SR): int(r["end"] * TTS_SR)], gap_s, w, gap_l]
    write_wav(out / "ab.wav", np.concatenate(ab), TTS_SR)
    with open(out / "script.tsv", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r['start']:.2f}\t{r['end']:.2f}\t{r['text']}\n")
    log(f"wrote {out}: cloned {len(clone)/TTS_SR/60:.1f} min, original {len(orig)/TTS_SR/60:.1f} min")
    return out
