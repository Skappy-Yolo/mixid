---
title: MixID
emoji: 🎧
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Identify every track in a long-form DJ mix.
---

# MixID — Hugging Face Space deployment

Public demo of [MixID](https://github.com/Skappy-Yolo/mixid) — a tool
that identifies every track in a DJ mix.

This Space runs the **Shazam-only fast pipeline** (no Demucs, no auto
Deep Scan). Sufficient for most public mixes; for the heavy stem-
separation pass, use the laptop deployment.

## What this Space exposes

- `/` — the PWA frontend
- `/jobs` — POST a file upload or URL to start a job
- `/jobs/{id}` — poll for status
- `/stats` — aggregate counter

## Configure your secrets

In your Space's "Settings → Variables and secrets", add:

```
ACOUSTID_API_KEY          (free from acoustid.org/api-key)
GEMINI_API_KEY            (optional, free from aistudio.google.com/app/apikey)
SPOTIFY_CLIENT_ID         (optional, for "Add to playlist" feature)
SPOTIFY_CLIENT_SECRET     (optional)
SPOTIFY_HOST_REFRESH_TOKEN (optional, see github.com/Skappy-Yolo/mixid)
```

Without these, MixID still works — it just skips the corresponding step
gracefully.

## Deploy

This Space rebuilds from the Dockerfile every push. After cloning the
upstream repo:

```bash
git remote add space https://huggingface.co/spaces/<your-username>/mixid
git push space main
```

The first build takes ~5-10 min (downloads dependencies + Whisper-tiny
model). After that, pushes deploy in ~1 min.

## Resource expectations

- RAM: ~1.5-2 GB during a pipeline run
- CPU: 2 vCPU (HF Spaces free tier)
- Runtime per 1-hour mix: ~30 min (Shazam path)
- Sleeps after 48h idle, wakes on first visit (~30s cold start)
