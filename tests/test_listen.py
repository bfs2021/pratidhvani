import json
import re

import numpy as np
import soundfile as sf

from voiceclone.compare import listening_results
from voiceclone.listen import write_listening_page


def bench_run(path, sentences, real_dir):
    path.mkdir(parents=True)
    takes = []
    for i, s in enumerate(sentences):
        for k in (0, 1):
            f = path / f"s{k}_{i}.wav"
            sf.write(f, np.zeros(800, np.float32), 16000)
            takes.append({"text": s, "seed": k, "path": str(f)})
    real = [{"text": "ನಿಜ", "path": str(p)} for p in sorted(real_dir.glob("*.wav"))]
    (path / "scores.json").write_text(json.dumps({"run": path.name, "spec_hash": "h", "takes": takes, "real": real}))
    return path


def test_page_is_blind_self_contained_and_scorable(tmp_path):
    real = tmp_path / "real"; real.mkdir()
    for i in range(3):
        sf.write(real / f"r{i}.wav", np.zeros(800, np.float32), 16000)
    sentences = [f"ವಾಕ್ಯ {i}" for i in range(6)]
    a, b = bench_run(tmp_path / "run_a", sentences, real), bench_run(tmp_path / "run_b", sentences, real)
    page, key_path = write_listening_page(a, b, tmp_path / "listen.html", check_every=4, log=lambda *x: None)
    html = page.read_text(encoding="utf-8")

    # nothing on the page says which take is which run
    for leak in ("run_a", "run_b", ".wav", str(tmp_path)):
        assert leak not in html
    assert not re.search(r"""src=["']https?:""", html)       # works offline
    key = json.loads(key_path.read_text(encoding="utf-8"))
    pairs = [t for t in key["trials"].values() if t["kind"] == "pair"]
    checks = [t for t in key["trials"].values() if t["kind"] == "check"]
    assert len(pairs) == 12 and len(checks) == 3
    assert all({t["1"], t["2"]} == {"A", "B"} for t in pairs)
    assert all("real" in (t["1"], t["2"]) for t in checks)
    assert html.count("data:audio/wav;base64,") == 2 * len(key["trials"])
    sides = {t["1"] for t in pairs}
    assert sides == {"A", "B"}                                 # A is not always on the same side

    # a listener who always picks run B and always spots the real clip
    picks = []
    for tid, t in key["trials"].items():
        want = "real" if t["kind"] == "check" else "B"
        side = "1" if t["1"] == want else "2"
        picks.append({"trial": tid, "voice": side, "natural": side})
    res = listening_results(key, [{"page_id": key["page_id"], "answers": picks}], a, b)
    assert res["questions"]["voice"]["b_win_rate"] == 1.0
    assert res["questions"]["voice"]["verdict"] == "B preferred"
    assert res["listeners"][0]["picked_real"]["voice"] == "3/3"
