"""Synthesize text in a reference speaker's voice (zero-shot or with a LoRA)."""
import time
from dataclasses import dataclass
from pathlib import Path

from voiceclone.audio import TTS_SR, load_wav, write_wav
from voiceclone.jsonl import write_json
from voiceclone.svara.tokens import EOS, PAD, build_prompt, token_budget
from voiceclone.text import syllables
from voiceclone.watermark import Watermark

DEFAULT_MODEL = "kenpath/svara-tts-voiceclone-beta"


@dataclass
class GenParams:
    temperature: float = 0.6
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    max_sec: float = 20.0
    min_sps: float = 2.0     # cap length at syllables/min_sps + 1.5 s
    max_sps: float = 6.0     # floor length at syllables/max_sps s


class Synthesizer:
    def __init__(self, model_dir: str = DEFAULT_MODEL, lora: str | None = None, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from voiceclone.device import default_dtype, pick_device
        from voiceclone.svara.codec import Codec

        self.device = device or pick_device()
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=default_dtype(self.device))
        model = model.to(self.device).eval()
        if lora:
            from peft import PeftModel

            # keep the adapter unmerged: merging fp32 LoRA deltas into bf16 weights rounds most of them away
            model = PeftModel.from_pretrained(model, lora).eval()
        self.model = model
        self.codec = Codec(self.device)
        self.watermark = Watermark()
        self._torch = torch

    def reference(self, wav_path) -> list[int]:
        return self.codec.encode(load_wav(wav_path, TTS_SR))

    def generate(self, text: str, ref_tokens: list[int] | None, ref_text: str | None,
                 seed: int = 0, p: GenParams = GenParams()):
        torch = self._torch
        torch.manual_seed(seed)
        ids = torch.tensor([build_prompt(self.tok, text, ref_tokens, ref_text)], device=self.device)
        n_syl = syllables(text)
        if n_syl == 0:
            import warnings

            warnings.warn(f"cannot count syllables in {text[:40]!r}; only the {p.max_sec:.0f} s cap applies")
        min_new, max_new = token_budget(n_syl, p.max_sec, p.min_sps, p.max_sps)
        with torch.inference_mode():
            # use_cache=True matters: the saved config has use_cache false, which
            # makes generate recompute the whole sequence per token (about 10x slower)
            gen = self.model.generate(ids, do_sample=True, temperature=p.temperature, top_p=p.top_p,
                                      repetition_penalty=p.repetition_penalty, max_new_tokens=max_new,
                                      min_new_tokens=min_new, use_cache=True, eos_token_id=EOS, pad_token_id=PAD)
        new = gen[0, ids.shape[1]:].tolist()
        if EOS in new:
            new = new[: new.index(EOS)]
        # every take is watermarked; there is deliberately no switch to turn it off
        return self.watermark.embed(self.codec.decode(new), TTS_SR), len(new)


def synthesize_to_dir(synth: Synthesizer, sentences: list[str], ref_wav, ref_text, out_dir,
                      seed: int = 0, p: GenParams = GenParams(), log=print) -> Path:
    """Write sample_XX.wav per sentence plus log.json {file,text,seconds,gen_time}."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ref_tokens = synth.reference(ref_wav)
    log(f"reference: {len(ref_tokens)//7} frames ({len(ref_tokens)/7/12:.1f}s), {len(ref_tokens)} tokens")
    rows = []
    for i, text in enumerate(sentences):
        t0 = time.time()
        wav, n = synth.generate(text, ref_tokens, ref_text, seed=seed + i, p=p)
        path = out / f"sample_{i:02d}.wav"
        write_wav(path, wav, TTS_SR)
        dt = time.time() - t0
        log(f"[{i}] {n} tokens -> {len(wav)/TTS_SR:.1f}s audio in {dt:.0f}s | {text[:60]}")
        rows.append({"file": path.name, "text": text, "seconds": round(len(wav) / TTS_SR, 2), "gen_time": round(dt, 1)})
    write_json(out / "log.json", rows)
    return out
