"""Verbatim transcription with AI4Bharat IndicConformer 600M (MIT).

The model is gated on Hugging Face: accept its terms and run `hf auth login`.
"""
from pathlib import Path

from voiceclone.jsonl import read_jsonl

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"


class IndicConformer:
    def __init__(self, model_id: str = MODEL_ID):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

    def transcribe(self, x, sr: int, lang: str = "kn", decoder: str = "rnnt") -> str:
        import torch
        import torchaudio

        w = torchaudio.functional.resample(torch.from_numpy(x).view(1, -1), sr, 16000)
        text = self.model(w, lang, decoder)
        text = text[0] if isinstance(text, (list, tuple)) else text
        return text.strip()


def transcribe_dir(vad_dir, lang: str = "kn", limit: int = 0, log=print, asr=None) -> Path:
    """Append {file, text} to vad_dir/transcripts.jsonl for every chunk not yet done (resumable).

    `asr` reuses an already loaded IndicConformer (the model is 600M parameters).
    """
    import json

    import soundfile as sf

    vad = Path(vad_dir)
    out = vad / "transcripts.jsonl"
    done = {r["file"] for r in read_jsonl(out)} if out.exists() else set()
    segs = read_jsonl(vad / "segments.jsonl")
    asr = asr or IndicConformer()
    with open(out, "a", encoding="utf-8") as f:
        for s in segs[: limit or len(segs)]:
            if s["file"] in done:
                continue
            x, sr = sf.read(str(vad / s["file"]), dtype="float32")
            text = asr.transcribe(x, sr, lang)
            f.write(json.dumps({"file": s["file"], "text": text}, ensure_ascii=False) + "\n"); f.flush()
            log(f"{s['file']} {s['duration']:.1f}s | {text}")
    return out
