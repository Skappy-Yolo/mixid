"""Tests for run.py decision gates (no audio / network needed)."""
from __future__ import annotations

import pytest

from mixid.pipeline import run as run_mod
from mixid.pipeline import local_demucs


@pytest.fixture
def demucs_available(monkeypatch):
    """Force local_demucs.is_available() True so we test the gate logic,
    not the environment."""
    monkeypatch.setattr(local_demucs, "is_available", lambda: True)
    monkeypatch.delenv("MIXID_DISABLE_DEMUCS", raising=False)


def test_auto_demucs_fires_on_clean_mix_via_absolute_count(demucs_available):
    # 10 unknown out of 40 segments = 25% — below the old >50% gate, but the
    # absolute >=3 gate catches it. This is the user's clean-mix failure case.
    pools = [object()] * 30
    unknowns = [(float(i * 60), float(i * 60 + 30)) for i in range(10)]
    assert run_mod._should_auto_demucs(pools, unknowns) is True


def test_auto_demucs_fires_on_high_fraction(demucs_available):
    pools = [object()] * 2
    unknowns = [(0.0, 30.0), (60.0, 90.0), (120.0, 150.0)]  # 3/5 = 60%
    assert run_mod._should_auto_demucs(pools, unknowns) is True


def test_auto_demucs_skips_near_perfect_mix(demucs_available):
    # Only 2 unknowns out of 40 — below both gates.
    pools = [object()] * 38
    unknowns = [(0.0, 30.0), (60.0, 90.0)]
    assert run_mod._should_auto_demucs(pools, unknowns) is False


def test_auto_demucs_disabled_by_env(demucs_available, monkeypatch):
    monkeypatch.setenv("MIXID_DISABLE_DEMUCS", "1")
    pools = [object()] * 5
    unknowns = [(float(i * 60), float(i * 60 + 30)) for i in range(10)]
    assert run_mod._should_auto_demucs(pools, unknowns) is False


def test_auto_demucs_false_when_unavailable(monkeypatch):
    monkeypatch.setattr(local_demucs, "is_available", lambda: False)
    monkeypatch.delenv("MIXID_DISABLE_DEMUCS", raising=False)
    pools = [object()] * 5
    unknowns = [(float(i * 60), float(i * 60 + 30)) for i in range(10)]
    assert run_mod._should_auto_demucs(pools, unknowns) is False


def test_auto_demucs_false_on_empty(demucs_available):
    assert run_mod._should_auto_demucs([], []) is False
