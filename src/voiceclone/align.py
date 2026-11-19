"""CTC forced alignment and greedy decoding with the MMS wav2vec2 ONNX model.

Uses `ctc-forced-aligner` (MIT code). Its MMS weights are CC-BY-NC, so treat
this as a data-preparation and evaluation tool, not part of a shipped model.

Word quality is reported as mean per-frame log-prob over the word's span
(0 is perfect; below about -2.5 usually means text and audio disagree).
"""
import json
import re

import numpy as np

from voiceclone.text import star_tokens, strip_stage_directions


class Aligner:
    def __init__(self):
        from ctc_forced_aligner import AlignmentSingleton

        a = AlignmentSingleton()
        self.model, self.tokenizer = a.alignment_model, a.alignment_tokenizer

    def emissions(self, y16: np.ndarray, batch_size: int = 1):
        from ctc_forced_aligner import generate_emissions

        return generate_emissions(self.model, y16.astype(np.float32), batch_size=batch_size)

    def align_tokens(self, emissions, stride, toks, txt) -> list[dict]:
        from ctc_forced_aligner import get_alignments, get_spans, postprocess_results

        segments, scores, blank = get_alignments(emissions, toks, self.tokenizer)
        return postprocess_results(txt, get_spans(toks, segments, blank), stride, scores)

    def align_text(self, y16: np.ndarray, text: str) -> list[dict]:
        """Word timings for `text` in `y16`; each word gets `pf`, its per-frame score."""
        toks, txt = star_tokens(text)
        if not toks:
            return []
        em, stride = self.emissions(y16)
        words = self.align_tokens(em, stride, toks, txt)
        for w in words:
            w["pf"] = w["score"] / max(1, round((w["end"] - w["start"]) * 1000 / stride))
        return words

    def greedy_decode(self, y16: np.ndarray) -> str:
        """What the recogniser hears, as a letters-only romanized string."""
        from ctc_forced_aligner import VOCAB_DICT

        id2ch = {v: k for k, v in VOCAB_DICT.items()}
        em, _ = self.emissions(y16)
        out, prev = [], -1
        for i in em[:, : len(VOCAB_DICT)].argmax(-1):   # last column is the aligner's <star>
            if i != prev and i not in (0, 1, 2, 3):
                out.append(id2ch.get(int(i), ""))
            prev = i
        return re.sub(r"[^a-z]", "", "".join(out))


def align_recording(aligner: Aligner, audio_path, utterances: list[dict], emissions_cache=None,
                    log=print) -> list[dict]:
    """Align a full human transcript to a full recording, utterance by utterance.

    Emissions are computed once for the whole file (optionally cached to .npz).
    Each utterance is aligned in a window starting just before the previous
    one ended (the native aligner cannot hold a 30-minute x full-text matrix),
    and never later than 1 s before its coarse timestamp, which guards against
    a previous runaway alignment.
    """
    from pathlib import Path

    from voiceclone.audio import load16

    cache = Path(emissions_cache) if emissions_cache else None
    if cache and cache.exists():
        z = np.load(cache); emissions, stride = z["emissions"], int(z["stride"])
    else:
        emissions, stride = aligner.emissions(load16(audio_path), batch_size=2)
        if cache:
            np.savez(cache, emissions=emissions, stride=stride)
    log(f"emissions {emissions.shape}, stride {stride} ms")

    fps = 1000 / stride
    total_s = emissions.shape[0] / fps
    by_utt = {u["idx"]: {**u, "words": []} for u in utterances}
    prev_end = 0.0
    for k, u in enumerate(utterances):
        toks, txt = star_tokens(strip_stage_directions(u["text"]))
        if not toks:
            continue
        nxt = utterances[k + 1]["start"] if k + 1 < len(utterances) else total_s
        w0 = max(0.0, min(prev_end - 0.3, u["start"] - 1.0)) if k else 0.0
        w1 = min(total_s, nxt + 5.0)
        try:
            words = aligner.align_tokens(emissions[int(w0 * fps):int(w1 * fps)], stride, toks, txt)
        except Exception as e:
            log(f"[{u['idx']}] FAILED {e}"); continue
        for w in words:
            frames = max(1, round((w["end"] - w["start"]) * fps))
            by_utt[u["idx"]]["words"].append({"text": w["text"], "start": round(w0 + w["start"], 3),
                                              "end": round(w0 + w["end"], 3),
                                              "score": round(w["score"] / frames, 3)})
        if words:
            prev_end = w0 + words[-1]["end"]
    res = list(by_utt.values())
    for u in res:
        ws = u["words"]
        if ws:
            log(f"[{u['idx']:2d}] {u['speaker'][:15]:15s} coarse={u['start']:5d}s "
                f"aligned={ws[0]['start']:7.2f}..{ws[-1]['end']:7.2f} words={len(ws):3d} "
                f"mean={np.mean([w['score'] for w in ws]):6.2f}")
    return res
