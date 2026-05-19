# Going public — Cloudflare Tunnel walkthrough (Windows)

Goal: anyone on the internet can hit your MixID server at a public HTTPS
URL without renting a server. Your laptop runs MixID; Cloudflare just
forwards encrypted traffic to it.

Free, no signup needed for the basic version, ~5 minutes total.

---

## What you need before starting

- MixID already installed and working locally (you can run `python -m mixid --serve` and load http://localhost:8000 in a browser).
- Windows 10 or 11 with admin rights (only needed to install cloudflared, not to run it).
- An internet connection (any home Wi-Fi).

## Step 1 — Install cloudflared

**Option A (recommended): winget**

Open PowerShell as Administrator and run:

```powershell
winget install --id Cloudflare.cloudflared
```

Hit `Y` if asked to accept the license. Done.

**Option B (manual): download the .exe**

1. Go to https://github.com/cloudflare/cloudflared/releases/latest
2. Download `cloudflared-windows-amd64.exe`
3. Rename it to just `cloudflared.exe`
4. Move it to `C:\Windows\System32\` (so Windows can find it from any terminal)

**Verify** (any terminal):

```powershell
cloudflared --version
```

You should see something like `cloudflared version 2025.x.x`. If the
command isn't found, close and reopen your terminal — PATH changes need a
fresh shell.

## Step 2 — Start MixID

Open a PowerShell terminal in the MixID folder:

```powershell
cd "$HOME\OneDrive\Documents\MixID"
.\.venv\Scripts\python.exe -m mixid --serve
```

You should see:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Don't close this terminal.** While the server is running, open http://localhost:8000/ in your browser — you should see the MixID page. If it loads, you're ready.

## Step 3 — Open the tunnel

Open a **second** PowerShell terminal (leave the first one running the server) and run:

```powershell
cloudflared tunnel --url http://localhost:8000
```

After about 10 seconds you'll see output like:

```
2026-05-19T12:34:56Z INF +--------------------------------------------------+
2026-05-19T12:34:56Z INF |  Your quick Tunnel has been created!             |
2026-05-19T12:34:56Z INF |  Visit it at (it may take some time to be       |
2026-05-19T12:34:56Z INF |  reachable):                                     |
2026-05-19T12:34:56Z INF |    https://random-words-1234.trycloudflare.com  |
2026-05-19T12:34:56Z INF +--------------------------------------------------+
```

**That `https://random-words-1234.trycloudflare.com` URL is your public MixID demo.** Open it on your phone, share it with friends, put it in your TikTok bio.

## Step 4 — Verify it works end-to-end

1. On your phone (different network if possible — turn off Wi-Fi and use cell data), open the public URL.
2. You should see the MixID PWA load.
3. Paste a short YouTube DJ mix URL and tap "Generate tracklist".
4. The job should start; check the laptop terminal — you'll see log lines from the server doing the work.
5. When the tracklist appears on the phone, the round-trip works.

## Step 5 — Don't close the terminals

While cloudflared runs in terminal #2 and `mixid --serve` runs in terminal #1, your public URL works. **If you close either:**

- Close the server terminal → URL returns "Bad Gateway" until you restart it
- Close cloudflared → URL stops working entirely (the tunnel dies)
- Close your laptop lid → service offline until you wake it

For a quick demo this is fine. For 24/7 availability see "Stable subdomain" below.

## When the URL changes (rerunning cloudflared)

Every time you re-run `cloudflared tunnel --url http://localhost:8000`, you get a NEW random URL. If you've shared a URL and want to keep it stable, see the next section.

## Optional: stable subdomain (e.g., mixid.yourdomain.com)

The free `*.trycloudflare.com` URLs change every restart. For a permanent URL you'll need:

1. A free Cloudflare account (sign up at https://dash.cloudflare.com/sign-up)
2. A domain on Cloudflare DNS (you can buy one for ~$10/yr or move an existing one)
3. Run `cloudflared tunnel login` and authenticate
4. Create a named tunnel: `cloudflared tunnel create mixid`
5. Add a CNAME in your Cloudflare DNS pointing `mixid.yourdomain.com` to the tunnel
6. Start the tunnel as a service so it runs in the background: `cloudflared service install`

Full walkthrough: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/create-remote-tunnel/

## Optional: keep your laptop awake while serving

Windows sleeps the machine on idle by default. To stop that while serving:

1. Open Settings → System → Power & battery
2. Screen and sleep → "When plugged in, put my device to sleep after" → Never
3. Optional: also set "When on battery" to Never if you'll demo on battery

Re-enable when you stop serving — running 24/7 wears the battery.

## Optional: register the Cloudflare URL with Spotify

If you've set up the Spotify "Add to playlist" feature (`python -m mixid.web.spotify_setup`), you'll want to register your Cloudflare URL as a redirect URI:

1. Go to https://developer.spotify.com/dashboard
2. Open your MixID app
3. "Edit Settings" → Redirect URIs
4. Add `https://your-cloudflare-url.trycloudflare.com/spotify/callback`
5. Save

If you're using ephemeral `*.trycloudflare.com` URLs that change on restart, you have two options: register the new URL every time (annoying), OR move to a stable subdomain (the "Optional" section above).

## Troubleshooting

- **"cloudflared: command not found"** — close and reopen the terminal. PATH only updates in new shells.
- **"Address already in use" on port 8000** — something else is using port 8000. Kill it or run MixID on a different port: `python -m mixid --serve --port 8765` then `cloudflared tunnel --url http://localhost:8765`.
- **Bad Gateway on the public URL** — the server died or you killed it. Restart with `python -m mixid --serve`.
- **First load is slow** — the tunnel needs ~10-30 sec to propagate after start. Wait, then refresh.
- **Phone uploads fail** — your phone may be on a different network than the laptop. The Cloudflare tunnel routes through Cloudflare's edge, so different networks are fine; the issue is usually the laptop's firewall. Check Windows Defender Firewall → Allow Python through.
