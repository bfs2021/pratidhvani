"""Denoise / restore bandwidth with ClearerVoice-Studio (Apache 2.0).

clearvoice pins numpy<2, so run this module in its own environment:
  python -m voiceclone.enhance.clearvoice IN_DIR OUT_DIR [--steps se] [--wet 0.9]
    se : MossFormer2_SE_48K speech enhancement (noise, some reverb). Default.
    sr : MossFormer2_SR_48K super-resolution (restores high frequencies).
         Off by default: on archival lecture audio it moved the median
         formants sharply (F2 1241 -> 801 Hz, F3 2280 -> 1265 Hz in the
         Vani-Harness ablation), and formants carry the accent.
  --wet  share of the enhanced signal; the rest is the original (dry).
         Vani-Harness uses 0.9 "to retain vocal warmth"; default 1.0.
Output is mono 24 kHz, peak -3 dBFS. Imports nothing heavy from voiceclone.
"""
import argparse
import tempfile
from pathlib import Path

MODELS = {"se": ("speech_enhancement", "MossFormer2_SE_48K"),
          "sr": ("speech_super_resolution", "MossFormer2_SR_48K")}


def blend(wet_sig, dry_sig, wet: float):
    """wet * enhanced + (1 - wet) * original, after matching lengths (models may pad)."""
    import numpy as np

    wet_sig = np.asarray(wet_sig, dtype=np.float32).reshape(-1)
    dry_sig = np.asarray(dry_sig, dtype=np.float32).reshape(-1)
    if len(wet_sig) != len(dry_sig):
        wet_sig = wet_sig[: len(dry_sig)] if len(wet_sig) > len(dry_sig) else np.pad(wet_sig, (0, len(dry_sig) - len(wet_sig)))
    return (wet * wet_sig + (1.0 - wet) * dry_sig).astype(np.float32)


def enhance_dir(inp, out, steps=("se",), wet: float = 1.0, log=print):
    import librosa
    import numpy as np
    import soundfile as sf
    from clearvoice import ClearVoice

    inp, out = Path(inp), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    models = {s: ClearVoice(task=MODELS[s][0], model_names=[MODELS[s][1]]) for s in steps}
    for f in sorted(inp.glob("*.wav")):
        with tempfile.TemporaryDirectory() as td:
            cur = Path(td) / "x.wav"
            dry = librosa.load(str(f), sr=48000, mono=True)[0]
            sf.write(cur, dry, 48000)
            for st in steps:
                res = models[st](input_path=str(cur), online_write=False)
                res = res[0] if isinstance(res, (list, tuple)) else res
                cur = Path(td) / f"{st}.wav"
                sf.write(cur, np.asarray(res).reshape(-1).astype(np.float32), 48000)
            if wet < 1.0:
                sf.write(cur, blend(sf.read(cur, dtype="float32")[0], dry, wet), 48000)
            y, _ = librosa.load(str(cur), sr=24000, mono=True)
            sf.write(out / f.name, y * (10 ** (-3 / 20) / (np.abs(y).max() or 1)), 24000, subtype="PCM_16")
            log(f.name)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m voiceclone.enhance.clearvoice")
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--steps", default="se", help="comma-separated: se, sr (sr shifts formants)")
    ap.add_argument("--wet", type=float, default=1.0, help="enhanced share of the output (0-1)")
    a = ap.parse_args(argv)
    if not 0.0 <= a.wet <= 1.0:
        ap.error("--wet must be between 0 and 1")
    enhance_dir(a.inp, a.out, tuple(a.steps.split(",")), a.wet)


if __name__ == "__main__":
    main()
