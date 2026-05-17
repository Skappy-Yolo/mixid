"""Tier-2 enrichment: free Colab/HF GPU, async, ~10-30 min per mix.

Invoked from notebooks/02_enrich_run.ipynb. Reads Tier-1 outputs,
re-runs unknowns through Demucs stems + Whisper lyrics + CLAP/FAISS +
ACRCloud, then emits an enriched tracklist with per-segment provenance.
"""
