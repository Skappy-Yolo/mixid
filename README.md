# MixID

**Identify every track in a long-form DJ mix.**

MixID ingests a 60–180 minute DJ mix — recorded at a party, ripped from YouTube/Mixcloud, or shared by another DJ — and returns an ordered tracklist with timestamps. Two-tier pipeline: fast local identification in under 2 minutes, optional cloud-GPU enrichment that runs while you make dinner.

## Status

Pre-alpha. Active development on `main`. See [the plan](https://github.com/Skappy-Yolo/mixid) for the staged build (~18 phases, ~40-60 commits).

## Why

There are three other DJ systems already in this toolkit: a YouTube downloader, a metadata indexer, and a playlist curator. None of them answer the question *"what songs are in this mix?"* MixID is that fourth piece.

## How it works (short version)

| Tier | Where it runs | Time | What it does |
|---|---|---|---|
| **Tier 1 — interactive** | Your laptop, CPU only | <2 min | URL shortcut → audio prep → hybrid segmentation → Chromaprint fingerprint with **pitch-shift sweep** → AcoustID remote → constrained LLM re-ranker → tracklist v1 |
| **Tier 2 — async enrichment** | Free Colab/HF GPU | ~10-30 min | Demucs vocal stems → vocal-fingerprint sweep → Whisper-small lyrics → Genius/lyrics.ovh → CLAP embeddings + FAISS → ACRCloud trial → HMM beam-search smoothing → LLM candidate-generator (validated) → tracklist v2 |

**Public APIs MixID matches against:** [AcoustID](https://acoustid.org/) (the ~50M-track open fingerprint database, free), [mixesdb.com](https://www.mixesdb.com/) (crowdsourced tracklists), [lyrics.ovh](https://lyrics.ovh/) + [Genius](https://genius.com/api-clients) (lyrics search), [ACRCloud](https://www.acrcloud.com/) free trial. You don't need any music files of your own — point MixID at any mix file or URL and it identifies tracks against these public sources.

**Optional speed boost for DJs with their own catalogs:** if you build a local fingerprint index over your music library (one-time, ~1 sec per track), MixID checks it before hitting AcoustID. Saves API calls and runs instantly. Skip the index build entirely if you don't have a library.

The pitch-shift sweep is the highest-ROI single change vs. naive fingerprinting (DJs pitch ±3-6% for beatmatching, which breaks Chromaprint silently).

## Honest accuracy targets

| Input type | Expected track-level recall |
|---|---|
| Own studio set (tracks in your library) | 90-95% |
| YouTube / Mixcloud mix (public catalog) | 70-80% |
| Phone party recording, clean | 60-72% |
| Phone party recording, noisy crowd | 35-50% |

Outputs flag every unidentified segment with its timestamp so you know exactly what to listen back to.

## Quick start

**Tier-1 (your laptop, CPU only):**

```bash
git clone https://github.com/Skappy-Yolo/mixid
cd mixid
pip install -e .
# Download fpcalc from https://github.com/acoustid/chromaprint/releases
# and place fpcalc.exe (or fpcalc on macOS/Linux) into bin/
cp .env.example .env  # then fill in your free AcoustID + Gemini keys
python -m mixid path/to/your/mix.mp3
```

Outputs land in `outputs/<run-id>/` as `tracklist.json`, `mix.m3u`, and `mix.txt`. Unknown segments are explicitly flagged with timestamps.

**Tier-2 (free Colab GPU, optional):**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skappy-Yolo/mixid/blob/main/notebooks/02_enrich_run.ipynb)

Open the notebook, upload your mix, run all cells. Demucs separates the vocal stem, then pitch-swept Chromaprint + AcoustID identifies tracks that the noisy full-mix fingerprint missed. The notebook outputs `enriched_tracklist.json`. Merge into your Tier-1 run with:

```bash
python -m mixid.enrich.merge outputs/<run-id>/ ~/Downloads/enriched_tracklist.json
```

## License

[MIT](LICENSE)
