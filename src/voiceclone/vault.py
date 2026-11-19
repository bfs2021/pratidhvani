"""Import cloning-vault segments from Vāṇī-Harness into this pipeline's chunk format.

Vāṇī-Harness (a separate restoration pipeline) separates vocals, removes crowd
noise and reverberation, enhances, and writes admitted 3-15 s segments plus a
JSON-lines index (`metadata.jsonl` or `vault_admitted_segments.jsonl`), one
object per segment with at least:
  segment_id, file_id, start_s, end_s, duration_s, admitted,
  spk_cosine, dnsmos_ovrl, vault_path, asr_transcript

Its transcripts are not imported: they come from Whisper, which hallucinated
on this speech (Turkish text and looping English on Kannada audio). Run
`voiceclone transcribe` on the imported folder for verbatim IndicConformer
text, then `build-clips` as usual. `vault_path` is usually an absolute path
on the machine that produced it, so files are looked up by name in
`vault_dir`.
"""
from pathlib import Path

from voiceclone.audio import TTS_SR, load_wav, peak_normalize, write_wav
from voiceclone.jsonl import read_jsonl, write_jsonl


def import_vault(index_path, vault_dir, out_dir, min_dnsmos: float | None = None,
                 min_spk: float | None = None, log=print) -> dict:
    """Copy admitted segments into out_dir as 24 kHz mono chunks + segments.jsonl.

    Each output row keeps where the audio came from (source file and its
    start/end in that recording) and the vault's quality scores.
    """
    vault, out = Path(vault_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, why = [], {}

    def drop(r):
        why[r] = why.get(r, 0) + 1

    for r in read_jsonl(index_path):
        if r.get("admitted") is False:
            drop("not_admitted"); continue
        if min_dnsmos is not None and float(r.get("dnsmos_ovrl", 0)) < min_dnsmos:
            drop("dnsmos"); continue
        if min_spk is not None and float(r.get("spk_cosine", 0)) < min_spk:
            drop("speaker"); continue
        src = vault / Path(r.get("vault_path") or f"{r['segment_id']}.wav").name
        if not src.exists():
            drop("missing_audio"); log(f"missing: {src}"); continue
        wav = peak_normalize(load_wav(src, TTS_SR))
        name = f"{r['segment_id']}.wav"
        write_wav(out / name, wav, TTS_SR)
        rows.append({"file": name, "start": round(float(r["start_s"]), 2), "end": round(float(r["end_s"]), 2),
                     "duration": round(len(wav) / TTS_SR, 2), "source": r.get("file_id"),
                     "dnsmos_ovrl": r.get("dnsmos_ovrl"), "spk_cosine": r.get("spk_cosine")})
    write_jsonl(out / "segments.jsonl", rows)
    return {"imported": len(rows), "minutes": sum(x["duration"] for x in rows) / 60, "dropped": why}
