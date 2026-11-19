from voiceclone.transcript import non_monotonic, parse_timed_transcript

SAMPLE = """[00:00:00] [Speaker A] : ಮೊದಲ ಸಾಲು
[46:31:00] [Speaker A] : ಎರಡನೇ ಸಾಲು

not a transcript line
[10:05:00] [Another Speaker] : ಮೂರನೇ
"""


def test_timestamp_is_seconds_then_minutes():
    # the export writes SS:MM, so [46:31:00] is 31 min 46 s, not 46 min 31 s
    utts, _ = parse_timed_transcript(SAMPLE)
    assert utts[1]["start"] == 31 * 60 + 46


def test_speakers_text_and_unparsed_lines():
    utts, bad = parse_timed_transcript(SAMPLE)
    assert [u["speaker"] for u in utts] == ["Speaker A", "Speaker A", "Another Speaker"]
    assert utts[0]["text"] == "ಮೊದಲ ಸಾಲು"
    assert [u["idx"] for u in utts] == [0, 1, 2]
    assert bad == ["not a transcript line"]


def test_non_monotonic_flags_backwards_timestamps():
    utts, _ = parse_timed_transcript(SAMPLE)
    # [10:05:00] is 5 min 10 s, earlier than the 31 min 46 s line before it
    assert non_monotonic(utts) == [2]
