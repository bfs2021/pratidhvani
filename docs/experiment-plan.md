# Experiment plan

What to try next, in order, and how to run each one so the result counts.
The rules are in [CONTRIBUTING.md](../CONTRIBUTING.md), and what was tried
so far is in [experiments/log.md](experiments/log.md).

## How an experiment runs

1. **Pick the next item below** and open a draft pull request with its
   experiment card filled in: the hypothesis and the one change.
2. **Make the change** in `ci/recipe.json`, in the code, or both. Explore
   locally as much as you like, but only the gate's run counts.
3. **Check locally:** `pytest` and `python ci/check_hygiene.py` must pass. Mark
   the pull request ready and add the `experiment` label.
4. **The owner reviews and adds `benchmark`.** The gate trains your recipe on
   the shared data, benchmarks it against main's run and posts the verdict.
5. **On a pass** (better on at least one metric, worse on none), the owner
   merges, and your recipe becomes main's. **On a fail**, add one line to the
   log on what you learnt and close the pull request.

The shared data (recordings, transcribed chunks, the benchmark's held-out
recordings) lives on the gate runner and is managed by the owner. Ask them
for a copy to explore with. Don't upload the recordings anywhere else: they
are a family archive.

## Backlog

Each item names its change, what should move on the gate, and its rough
cost on the runner.

| # | Experiment | The one change | Expect | Cost |
|---|---|---|---|---|
| 0 | **Baseline** (owner) | Score main's recipe once | Sets the numbers every other run is compared with | 1 run |
| 1 | **Clean close-mic data, verbatim** | The owner prepares DOC-0004 (`prepare`, IndicConformer text). Add its chunk folder to the recipe's `build-clips` | Similarity up; CER no worse. EXP-003/007 saw the gain with Whisper text, and this tests it with verbatim text | 1 run |
| 2 | **Pitch gate from his own clips** | Add `--f0-from-enroll` to `build-clips` | Similarity up when hall recordings are in the set; no change without them | 1 run |
| 3 | **More of his speech from the hall recordings** | New `diarize` step (pyannote, MIT) keeping only his turns, then `build-clips` | More training minutes at the same purity; similarity up | Code, then 1 run |
| 4 | **Dereverberation** | Same recipe, chunks enhanced with MossFormer2 (`prepare --enhance-python`) versus not | Answers whether reverb broke EXP-005. CER should fall if it helps | 1 run |
| 5 | **Pick the checkpoint by held-out loss** | `train` computes validation loss on held-out recordings; the recipe uses the best checkpoint | Avoids the overfitting that low training loss hides | Code, then 1 run |
| 6 | **Capacity** | Rank 32 (and later epochs and learning rate), one at a time | Only worth it once 1-4 give 20+ minutes of clean speech | 1 run each |
| 7 | **Best of 3 in the product** | Studio previews generate 3 takes and keep the best by similarity and CER (`best-of`) | Better previews without retraining; checked by listening, not the gate | Code only |

Items 1, 2 and 4 are recipe changes and can go in order. Items 3 and 5 are
code first: one pull request for the new step, with tests, which the gate
checks for "no worse". Then an experiment pull request uses it.

## Claims about the accent

The gate measures voice similarity and intelligibility, not accent. For an
accent claim, generate the A/B page from the gate's two runs
(`voiceclone listen-page`). Have at least three native speakers of the dialect
answer it, and attach `voiceclone compare --answers ...` to the pull request.

## Not worth repeating

- Zero-shot conditioning alone (EXP-001/002): it can't hold this voice.
- Choosing the conditioning clip or the reference by its score against the
  evaluation target: that is the leak the benchmark refuses.
- Judging by training loss, or by audio after loudness normalisation.
