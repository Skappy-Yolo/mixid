# Deployment — honest comparison

You have three realistic paths to put MixID on a public URL. Each has
trade-offs. Pick based on what you can tolerate, not what sounds cool.

| Path | Cost | Setup time | Always on? | Full Demucs? | Best for |
|---|---|---|---|---|---|
| **Cloudflare Tunnel** | $0 | 5 min | only when laptop awake | ✅ yes | Your own use, demo to friends, you don't mind your laptop running |
| **Hugging Face Spaces** | $0 | 15 min | yes (sleeps after 48h idle, wakes on visit) | ❌ no | Public portfolio demo, always-on URL |
| **Fly.io paid tier** | ~$5/mo | 10 min | yes | ✅ slow (CPU) | You're OK with paying a little for clean ops |

If you only want to do ONE: **Hugging Face Spaces** for the public TikTok-linkable URL.

---

## Path 1: Cloudflare Tunnel (your laptop)

See [CLOUDFLARE_TUNNEL.md](./CLOUDFLARE_TUNNEL.md) for the full walkthrough.

**TL;DR**:
```powershell
# Terminal 1
cd "$HOME\OneDrive\Documents\MixID"
.\.venv\Scripts\python.exe -m mixid --serve

# Terminal 2
cloudflared tunnel --url http://localhost:8000
```

Use the printed `https://*.trycloudflare.com` URL.

---

## Path 2: Hugging Face Spaces (recommended public demo)

Free, always-on (sleeps after 48h idle, wakes on first visit), 16 GB
RAM, no credit card. Trade-off: cloud-only deployment skips Demucs (no
auto Deep Scan). Shazam-only pipeline still gets ~25-35 tracks on a
typical 1-hour DJ mix.

### Step 1 — Create a Space

1. Go to https://huggingface.co — sign up if you haven't.
2. Top-right → New → Space.
3. Owner: your username. Space name: `mixid`. License: MIT.
4. SDK: **Docker**. Visibility: Public.
5. Create.

You now have an empty Space at `https://huggingface.co/spaces/<your-username>/mixid`.

### Step 2 — Push the code

```bash
cd "$HOME\OneDrive\Documents\MixID"
git remote add space https://huggingface.co/spaces/<your-username>/mixid
git push space main
```

Note: the Space reads the `huggingface_space/` subdirectory by default
because the README and Dockerfile live there. If you want it at the repo
root, copy those two files up.

### Step 3 — Configure secrets

On your Space's page → Settings → Variables and secrets → Add new secret:

| Name | Required? | Where to get |
|---|---|---|
| `ACOUSTID_API_KEY` | Recommended | https://acoustid.org/api-key (free) |
| `GEMINI_API_KEY` | Optional | https://aistudio.google.com/app/apikey (free) |
| `SPOTIFY_CLIENT_ID` | Optional | https://developer.spotify.com/dashboard |
| `SPOTIFY_CLIENT_SECRET` | Optional | same |
| `SPOTIFY_HOST_REFRESH_TOKEN` | Optional | run `python -m mixid.web.spotify_setup` locally first |

Without any of these, MixID still loads — it just skips the
corresponding identification step.

### Step 4 — Wait for the first build

The first push triggers a Docker build that takes 5-10 min (downloads
dependencies, sets up ffmpeg + libchromaprint, fetches the Whisper-tiny
model). You can watch progress in the "Logs" tab.

Once it says "Running", your public URL is live:
```
https://<your-username>-mixid.hf.space
```

That's the URL for TikTok, Substack, LinkedIn — wherever.

### Step 5 — Spotify redirect URI (only if using Spotify export)

If you set `SPOTIFY_HOST_REFRESH_TOKEN`, register the HF Space URL as a
redirect URI in your Spotify Developer dashboard:

```
https://<your-username>-mixid.hf.space/spotify/callback
```

---

## Path 3: Fly.io paid tier ($5/mo)

Skip this unless you have a specific reason. Fly's free tier (256 MB)
won't run MixID. Their cheapest tier with enough RAM is ~$5/month.

If you do go this route:

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Launch (from MixID repo root)
fly launch --no-deploy

# Edit fly.toml — set memory to 2048MB minimum
# Deploy
fly deploy
```

You'll need to add the same env vars (`fly secrets set ACOUSTID_API_KEY=...`).
This setup is barely faster than HF Spaces for similar functionality.

---

## What you lose in the cloud deploy

The HF Spaces deployment intentionally **disables Demucs / auto Deep
Scan** (env var `MIXID_DISABLE_DEMUCS=1`). The reasons:

1. Demucs CPU runs are 2-4 hours per mix. A single user would block the
   Space's single concurrent slot for hours.
2. Demucs model is 2GB on disk and uses 3-4 GB RAM during inference —
   tight even at 16 GB total.
3. Heavy ML on free Spaces also burns through HF's compute allotment,
   risking the Space being throttled.

If a user wants the full Deep Scan pass, they can either:
- Install MixID locally (`pip install -e .[enrich]`) and run it themselves
- Use your Cloudflare-Tunnel-to-laptop demo if you have one running

You can mention both options in your Substack / TikTok captions.
