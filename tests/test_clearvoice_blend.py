import numpy as np

from voiceclone.enhance.clearvoice import blend, main


def test_blend_mixes_enhanced_and_original():
    out = blend(np.ones(4), np.zeros(4), 0.9)
    assert np.allclose(out, 0.9)


def test_blend_matches_lengths_when_the_model_pads_or_trims():
    dry = np.zeros(5)
    assert len(blend(np.ones(7), dry, 0.5)) == 5
    assert len(blend(np.ones(3), dry, 0.5)) == 5


def test_super_resolution_is_not_on_by_default(monkeypatch):
    # SR moved formants by hundreds of Hz on archival audio; it must be opt-in
    seen = {}
    monkeypatch.setattr("voiceclone.enhance.clearvoice.enhance_dir",
                        lambda inp, out, steps, wet: seen.update(steps=steps, wet=wet))
    main(["in", "out"])
    assert seen == {"steps": ("se",), "wet": 1.0}
