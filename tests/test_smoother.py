"""Gap-fill smoother tests — must rescue agreeing neighbors, never invent."""
from __future__ import annotations

from mixid.pipeline import smoother
from mixid.pipeline.reranker import RerankedResult


def _r(idx, start, end, artist, title, score=0.9, source="acoustid"):
    return RerankedResult(
        segment_index=idx,
        start_sec=start,
        end_sec=end,
        artist=artist,
        title=title,
        score=score,
        source=source,
    )


def test_fills_gap_when_both_neighbors_agree():
    reranked = [
        _r(0, 0, 30, "Burna Boy", "Last Last"),
        _r(2, 60, 90, "Burna Boy", "Last Last"),
    ]
    unknowns = [(30.0, 60.0)]
    out, rem = smoother.smooth_gaps(reranked, unknowns)
    assert rem == []
    assert len(out) == 3
    filled = [r for r in out if r.start_sec == 30.0][0]
    assert filled.artist == "Burna Boy" and filled.title == "Last Last"
    assert filled.source == "smoothed:acoustid"


def test_does_not_fill_when_neighbors_disagree():
    reranked = [
        _r(0, 0, 30, "Burna Boy", "Last Last"),
        _r(2, 60, 90, "Asake", "Lonely at the Top"),
    ]
    unknowns = [(30.0, 60.0)]
    out, rem = smoother.smooth_gaps(reranked, unknowns)
    assert rem == [(30.0, 60.0)]  # gap stays unknown
    assert len(out) == 2  # nothing added


def test_does_not_fill_huge_gaps():
    reranked = [
        _r(0, 0, 30, "A", "X"),
        _r(2, 200, 230, "A", "X"),
    ]
    unknowns = [(30.0, 200.0)]
    out, rem = smoother.smooth_gaps(reranked, unknowns, max_gap_secs=60.0)
    # 170-sec gap is too long — that's two tracks with a break
    assert rem == [(30.0, 200.0)]
    assert len(out) == 2


def test_does_not_fill_at_track_edges():
    """A gap with only ONE neighbor (start or end of mix) can't be filled."""
    reranked = [_r(0, 60, 90, "A", "X")]
    unknowns = [(0.0, 60.0), (90.0, 120.0)]
    out, rem = smoother.smooth_gaps(reranked, unknowns)
    assert rem == unknowns
    assert len(out) == 1


def test_filled_score_is_min_of_neighbors():
    reranked = [
        _r(0, 0, 30, "A", "X", score=0.95),
        _r(2, 60, 90, "A", "X", score=0.72),
    ]
    out, _ = smoother.smooth_gaps(reranked, [(30.0, 60.0)])
    filled = [r for r in out if r.start_sec == 30.0][0]
    assert filled.score == 0.72  # be honest about confidence


def test_default_gap_rescues_up_to_segment_cap():
    """The default max_gap (90s, matching MAX_SEGMENT_GAP_SECS) rescues a
    75s sandwiched gap that the old 60s default would have stranded."""
    reranked = [
        _r(0, 0, 30, "A", "X"),
        _r(2, 105, 135, "A", "X"),
    ]
    unknowns = [(30.0, 105.0)]  # 75s gap — between the old 60 and new 90 default
    out, rem = smoother.smooth_gaps(reranked, unknowns)  # no explicit max_gap
    assert rem == []
    filled = [r for r in out if r.start_sec == 30.0][0]
    assert filled.artist == "A" and filled.title == "X"
