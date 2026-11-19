"""LoRA fine-tune of the Svara voice-clone model on one speaker.

Each example mirrors the inference prompt ("reference-swap"): the reference
is another clip of the same speaker, and loss applies only to the target
audio block. Batch size 1 with gradient accumulation fits a 16 GB laptop.

The final adapter is saved in the output folder itself; intermediate ones
in checkpoint-NNNNN/ subfolders (about 100 MB each at rank 16). Any of them
can be passed to `synth --lora`.
"""
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from voiceclone.audio import TTS_SR, load_wav
from voiceclone.jsonl import read_jsonl, write_json
from voiceclone.svara.synth import DEFAULT_MODEL
from voiceclone.svara.tokens import training_example

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainParams:
    epochs: int = 3
    lr: float = 2e-4
    rank: int = 16
    max_ref_sec: float = 4.5
    max_tgt_sec: float = 10.5
    grad_accum: int = 4
    holdout: int = 6          # clips kept out of training for evaluation
    max_steps: int = 0
    seed: int = 0
    init_adapter: str | None = None   # continue training a saved LoRA
    save_every: int = 50      # optimizer steps between kept checkpoints; 0 = final only


def checkpoint_dir(out_dir, step: int) -> Path:
    """Where the adapter after `step` optimizer steps is kept. Each checkpoint
    gets its own folder, so an earlier one that sounded better is never lost."""
    return Path(out_dir) / f"checkpoint-{step:05d}"


def encode_clips(clips_dir, clips: list[dict], device) -> dict:
    """SNAC tokens per clip, cached in clips_dir/snac_tokens.json."""
    from voiceclone.svara.codec import Codec

    cache_path = Path(clips_dir) / "snac_tokens.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    missing = [c for c in clips if c["file"] not in cache]
    if missing:
        codec = Codec(device)
        for c in missing:
            cache[c["file"]] = codec.encode(load_wav(Path(clips_dir) / c["file"], TTS_SR))
        json.dump(cache, open(cache_path, "w"))
    return cache


MIN_REFS = 8   # distinct reference clips the reference swap draws from


def reference_pool(train: list[dict], max_ref_sec: float, log=print) -> list[dict]:
    """Clips that can serve as the prompt's reference: those up to `max_ref_sec`.

    Clips cut at pauses (`vad`, `prepare`, the studio) are mostly 5-10 s, so a
    small set can have one short clip or none, and every pair needs a reference
    other than its target. With fewer than MIN_REFS short clips, the shortest
    MIN_REFS clips are used instead.
    """
    if len(train) < 2:
        raise ValueError(f"training needs at least 2 clips, got {len(train)}")
    refs = [c for c in train if c["duration"] <= max_ref_sec]
    if len(refs) < MIN_REFS:
        short = len(refs)
        refs = sorted(train, key=lambda c: c["duration"])[:MIN_REFS]
        log(f"only {short} training clips are {max_ref_sec:g} s or shorter; using the {len(refs)} shortest "
            f"as references (up to {refs[-1]['duration']:.1f} s)")
    return refs


def finetune(clips_dir, out_dir, model_dir: str = DEFAULT_MODEL, p: TrainParams = TrainParams(), log=print):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from voiceclone.device import default_dtype, pick_device

    random.seed(p.seed); torch.manual_seed(p.seed)
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(model_dir)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    clips = [c for c in read_jsonl(Path(clips_dir) / "metadata.jsonl") if c["duration"] <= p.max_tgt_sec]
    random.shuffle(clips)
    holdout, train = clips[: p.holdout], clips[p.holdout:]
    write_json(out / "holdout.json", holdout)
    cache = encode_clips(clips_dir, clips, device)
    log(f"train clips={len(train)} holdout={len(holdout)} train audio={sum(c['duration'] for c in train)/60:.1f} min")
    refs = reference_pool(train, p.max_ref_sec, log)

    class Pairs(Dataset):
        def __len__(self):
            return len(train)

        def __getitem__(self, i):
            tgt = train[i]
            ref = random.choice([r for r in refs if r["file"] != tgt["file"]])
            ids, labels = training_example(tok, ref["text"], cache[ref["file"]], tgt["text"], cache[tgt["file"]])
            return torch.tensor(ids), torch.tensor(labels)

    dl = DataLoader(Pairs(), batch_size=1, shuffle=True,
                    collate_fn=lambda b: tuple(torch.stack(x) for x in zip(*b)))

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=default_dtype(device)).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    if p.init_adapter:
        model = PeftModel.from_pretrained(model, p.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=p.rank, lora_alpha=2 * p.rank, lora_dropout=0.05, bias="none",
                                                 target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    for prm in model.parameters():          # LoRA weights in fp32; base stays bf16
        if prm.requires_grad:
            prm.data = prm.data.float()
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=p.lr, weight_decay=0.0)
    total = p.max_steps or (p.epochs * len(dl)) // p.grad_accum
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * max(0.05, 1 - s / max(1, total)))

    model.train()
    step, micro, t0, losses, saved = 0, 0, time.time(), [], []
    for epoch in range(p.epochs):
        for ids, labels in dl:
            ids, labels = ids.to(device), labels.to(device)
            # Project to the vocabulary only where a labelled token is predicted:
            # full-sequence fp32 logits over a 157k vocab cost several GB per step.
            hidden = model.base_model.model.model(input_ids=ids).last_hidden_state
            tgt = labels[:, 1:]
            mask = tgt != -100
            logits = model.base_model.model.lm_head(hidden[:, :-1][mask]).float()
            loss = torch.nn.functional.cross_entropy(logits, tgt[mask])
            (loss / p.grad_accum).backward()
            del hidden, logits
            losses.append(loss.item()); micro += 1
            if micro % p.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if device == "mps":
                torch.mps.empty_cache()
            if step % 2 == 0 or step == 1:
                el = time.time() - t0
                log(f"ep{epoch} step {step}/{total} loss {np.mean(losses[-p.grad_accum*5:]):.3f} "
                    f"seq {ids.shape[1]} {el/step:.1f}s/step eta {(total-step)*el/step/60:.0f}min")
            if p.save_every and step % p.save_every == 0:
                model.save_pretrained(checkpoint_dir(out, step)); saved.append(step)
            if p.max_steps and step >= p.max_steps:
                break
        else:
            continue
        break
    model.save_pretrained(out)
    write_json(out / "train_log.json", {"losses": losses, "steps": step, "params": asdict(p),
                                        "clips": str(clips_dir), "model": model_dir,
                                        "checkpoints": [checkpoint_dir(out, k).name for k in saved]})
    log(f"saved {out} final loss {np.mean(losses[-20:]):.3f}")
    return out
