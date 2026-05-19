# Exposing the local MixID server publicly via Cloudflare Tunnel

Goal: anyone on the internet can hit your local MixID server at a public
HTTPS URL without you renting any cloud infrastructure.

## 1. Install `cloudflared`

**Windows (recommended via winget):**

```powershell
winget install --id Cloudflare.cloudflared
```

Or download the latest release directly from
<https://github.com/cloudflare/cloudflared/releases> — pick the
`cloudflared-windows-amd64.exe`, rename to `cloudflared.exe`, and put it
on your PATH (e.g. drop into `C:\Windows\System32\`).

Verify:

```bash
cloudflared --version
```

## 2. Start the MixID server

In the MixID repo, activate the venv and run:

```powershell
cd "$HOME\OneDrive\Documents\MixID"
.\.venv\Scripts\python.exe -m mixid --serve
```

You should see uvicorn announce `http://0.0.0.0:8000`. Confirm it works
locally first: open <http://localhost:8000/> in your browser. You should
see the MixID PWA.

## 3. Expose it publicly

In a second terminal, run:

```bash
cloudflared tunnel --url http://localhost:8000
```

After a couple of seconds, cloudflared prints a public URL like
`https://random-words-1234.trycloudflare.com`. Share that URL —
**that's your public MixID demo.** No signup, no DNS, no port-forwarding.

The tunnel is ephemeral; restarting cloudflared gives you a different
URL. For a stable subdomain (e.g. `mixid.your-domain.dev`), sign in
with `cloudflared tunnel login`, create a named tunnel, and add a CNAME
in your Cloudflare account — see the
[Cloudflare Tunnel docs](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/create-remote-tunnel/)
for that path.

## 4. Things to keep in mind

- **Laptop must stay awake.** When it sleeps, the tunnel drops. For a
  demo, run with the lid open. For a 24/7 service, disable sleep on AC
  power: `Settings → System → Power → Screen and sleep → Never`.
- **One concurrent pipeline run at a time.** The server submits to a
  single-worker thread pool. Extra requests wait in the FastAPI event
  loop. Fine for early traffic, queue if it goes viral.
- **Bandwidth.** Cloudflare's free tunnel has no published bandwidth
  cap for personal use, but uploads + downloads of mix audio count.
  Watch your home network plan if this gets big.
- **Privacy.** MixID's web tier deletes uploaded audio immediately
  after processing. Only an aggregate counter (`~/.mixid/stats.db`)
  persists — no IPs, no titles, no PII. Cloudflare may keep request
  logs at their edge; if that's a concern, document it in your
  Substack post.
