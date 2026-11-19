from voiceclone.train import TrainParams, checkpoint_dir


def test_each_checkpoint_gets_its_own_folder(tmp_path):
    # earlier adapters must survive later saves, and sort in training order
    dirs = [checkpoint_dir(tmp_path, s) for s in (50, 100, 1000)]
    assert len(set(dirs)) == 3
    assert sorted(d.name for d in dirs) == [d.name for d in dirs]
    assert all(d.parent == tmp_path for d in dirs)


def test_checkpoints_kept_by_default():
    assert TrainParams().save_every > 0


def test_reference_pool_keeps_short_clips_when_there_are_enough():
    # the reference results trained this way; the fallback must not change them
    from voiceclone.train import reference_pool

    train = [{"file": f"c{i}", "duration": d} for i, d in enumerate([3.0, 4.0, 4.4] * 4 + [8.0, 9.0])]
    assert all(r["duration"] <= 4.5 for r in reference_pool(train, 4.5, log=lambda *a: None))


def test_reference_pool_falls_back_to_the_shortest_clips():
    # pause-cut clips are mostly 5-10 s: one short clip used to leave no reference to pair with
    from voiceclone.train import MIN_REFS, reference_pool

    train = [{"file": f"c{i}", "duration": 4.1 + i * 0.3} for i in range(30)]
    msgs = []
    refs = reference_pool(train, 4.5, log=msgs.append)
    assert len(refs) == MIN_REFS and refs[0]["duration"] == 4.1 and "shortest" in msgs[0]
    for tgt in train:   # every target has a different clip to pair with
        assert [r for r in refs if r["file"] != tgt["file"]]


def test_reference_pool_needs_two_clips():
    import pytest

    from voiceclone.train import reference_pool

    with pytest.raises(ValueError, match="at least 2"):
        reference_pool([{"file": "a", "duration": 3.0}], 4.5)
