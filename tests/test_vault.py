"""Importing another pipeline's vault: its audio is wanted, its transcripts are not,
and its absolute VM paths must not matter."""
import json

import numpy as np
import soundfile as sf

from voiceclone.jsonl import read_jsonl
from voiceclone.vault import import_vault


def make_vault(tmp_path):
    vault = tmp_path / "vault"; vault.mkdir()
    rows = []
    for i, (admitted, dns, spk) in enumerate([(True, 2.7, 0.44), (True, 2.2, 0.50), (False, 2.9, 0.20),
                                              (True, 2.8, 0.30)]):
        sid = f"talk_seg_{i:04d}"
        sf.write(vault / f"{sid}.wav", 0.1 * np.ones(48000 * 4, np.float32), 48000)
        rows.append({"segment_id": sid, "file_id": "talk", "start_s": 10.0 * i, "end_s": 10.0 * i + 4,
                     "duration_s": 4.0, "admitted": admitted, "dnsmos_ovrl": dns, "spk_cosine": spk,
                     "asr_transcript": "Ya da öyle bir paraya koyduğuna",
                     "vault_path": f"/Volumes/My Shared Files/VaniHarness/data/{sid}.wav"})
    with open(tmp_path / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return vault


def test_imports_admitted_audio_at_tts_rate_from_a_moved_vault(tmp_path):
    vault = make_vault(tmp_path)
    s = import_vault(tmp_path / "metadata.jsonl", vault, tmp_path / "out", log=lambda *a: None)
    rows = read_jsonl(tmp_path / "out" / "segments.jsonl")
    assert s["imported"] == 3 and s["dropped"] == {"not_admitted": 1}
    wav, sr = sf.read(tmp_path / "out" / rows[0]["file"])
    assert sr == 24000 and abs(len(wav) / sr - 4.0) < 0.01
    assert abs(np.abs(wav).max() - 10 ** (-3 / 20)) < 1e-3          # same peak level as our chunks
    assert rows[1]["start"] == 10.0 and rows[1]["source"] == "talk"  # provenance in source time


def test_vault_transcripts_are_not_carried_over(tmp_path):
    vault = make_vault(tmp_path)
    import_vault(tmp_path / "metadata.jsonl", vault, tmp_path / "out", log=lambda *a: None)
    assert not (tmp_path / "out" / "transcripts.jsonl").exists()
    assert all("text" not in r for r in read_jsonl(tmp_path / "out" / "segments.jsonl"))


def test_optional_quality_and_speaker_gates(tmp_path):
    vault = make_vault(tmp_path)
    s = import_vault(tmp_path / "metadata.jsonl", vault, tmp_path / "out", min_dnsmos=2.5, min_spk=0.4,
                     log=lambda *a: None)
    assert s["imported"] == 1 and s["dropped"] == {"not_admitted": 1, "dnsmos": 1, "speaker": 1}
