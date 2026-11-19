"""Rank candidate recordings for training: is it the speaker, and how clean is it?

Six 20 s excerpts spread across each file are scored on
  sim    : ECAPA similarity to the speaker centroid (identity, other speakers),
  snr    : speech level minus pause floor, dB (noise),
  bw     : 95% spectral roll-off, Hz (muffled vs open),
  speech : fraction of frames above the pause floor (density).
"""
from pathlib import Path

import numpy as np

AUDIO_EXT = (".mp3", ".wav", ".clips")


def rank_recordings(paths, encoder, speaker_centroid, log=print) -> list[dict]:
    import librosa

    files = [p for d in map(Path, paths) for p in (sorted(d.rglob("*")) if d.is_dir() else [d])
             if p.is_file() and p.suffix.lower() in AUDIO_EXT]
    rows = []
    for f in files:
        dur = librosa.get_duration(path=str(f))
        sims, snrs, bws, dens = [], [], [], []
        for frac in np.linspace(0.1, 0.9, 6):
            y, sr = librosa.load(str(f), sr=16000, offset=float(frac * dur), duration=20.0)
            if len(y) < sr * 5:
                continue
            db = librosa.amplitude_to_db(librosa.feature.rms(y=y, frame_length=320, hop_length=320)[0] + 1e-6)
            floor, speech = np.percentile(db, 10), np.percentile(db, 90)
            sims.append(float(np.dot(encoder.embed(y), speaker_centroid)))
            snrs.append(float(speech - floor))
            bws.append(float(np.median(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95))))
            dens.append(float((db > floor + 15).mean()))
        r = {"file": str(f), "minutes": round(dur / 60, 1), "sim": round(float(np.median(sims)), 3),
             "sim_min": round(float(np.min(sims)), 3), "snr": round(float(np.median(snrs)), 1),
             "bw": int(np.median(bws)), "speech": round(float(np.median(dens)), 2)}
        rows.append(r)
        log(f"{Path(r['file']).name[:44]:44s} {r['minutes']:5.1f}m sim {r['sim']:.2f} (min {r['sim_min']:.2f}) "
            f"snr {r['snr']:4.1f} bw {r['bw']:5d} speech {r['speech']:.2f}")
    return rows
