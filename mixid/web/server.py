"""FastAPI web app — wraps `mixid.pipeline.run.run` for browser/PWA use.

Endpoints:
  GET  /                  → serves the PWA single-page app
  GET  /stats             → returns total mixes processed (no PII)
  POST /jobs              → accepts file upload OR url; returns {job_id}
  GET  /jobs/{job_id}     → returns status + result when ready
  GET  /static/*          → PWA assets (manifest, service worker, icons)

The pipeline can take 30+ min for a 1-hour mix, so we run it in a
background thread and the frontend polls /jobs/{job_id} every few sec.

Single concurrent job by design — the CPU pipeline serializes naturally,
and a queue is not justified for laptop-scale traffic. If multiple
clients hit /jobs at once, they queue cooperatively.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from mixid.pipeline import run as pipeline_run
from mixid.web import mirror_search, spotify_export, stats, url_input

log = logging.getLogger(__name__)


# Single shared thread executor — pipeline is heavy CPU-bound work
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mixid-pipeline")


@dataclass
class Job:
    id: str
    status: str = "pending"            # pending | running | done | error
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    source_kind: str = "file"          # 'file' or 'url'
    source_label: str = ""              # filename or URL for display
    with_demucs: bool = False
    result: dict | None = None
    error: str | None = None


# Job registry — in-memory is fine for a laptop deploy
_JOBS: dict[str, Job] = {}

# Mirror-search cache: normalized URL → (timestamp, payload). 1 hour TTL,
# cap of 200 entries. Sweep on insert.
_MIRRORS_CACHE: dict[str, tuple[float, dict]] = {}
_MIRRORS_TTL_SEC = 3600.0
_MIRRORS_CACHE_MAX = 200


def _normalize_mirror_url(url: str) -> str:
    """Strip Spotify's ?si= share tokens and Apple's locale prefix so the same
    logical track from different share links collapses to one cache entry."""
    import urllib.parse as _up

    try:
        p = _up.urlsplit(url.strip())
    except Exception:
        return url
    host = (p.netloc or "").lower()
    path = p.path or ""
    if "music.apple.com" in host:
        # /us/album/... -> /album/...   (drop two-letter locale prefix)
        parts = path.split("/", 2)
        if len(parts) >= 3 and len(parts[1]) == 2 and parts[1].isalpha():
            path = "/" + parts[2]
        query = ""  # apple share params don't change identity
    elif "spotify" in host:
        query = ""  # ?si=... is share tracking; identity is in the path
    else:
        query = p.query
    return _up.urlunsplit((p.scheme.lower(), host, path, query, ""))


def _mirror_cache_get(url: str) -> dict | None:
    norm = _normalize_mirror_url(url)
    hit = _MIRRORS_CACHE.get(norm)
    if not hit:
        return None
    if time.time() - hit[0] > _MIRRORS_TTL_SEC:
        _MIRRORS_CACHE.pop(norm, None)
        return None
    return hit[1]


def _mirror_cache_put(url: str, payload: dict) -> None:
    now = time.time()
    norm = _normalize_mirror_url(url)
    _MIRRORS_CACHE[norm] = (now, payload)
    # Sweep expired
    if len(_MIRRORS_CACHE) > _MIRRORS_CACHE_MAX:
        # Drop expired first
        expired = [k for k, (t, _) in _MIRRORS_CACHE.items() if now - t > _MIRRORS_TTL_SEC]
        for k in expired:
            _MIRRORS_CACHE.pop(k, None)
        # If still over cap, drop oldest by timestamp
        if len(_MIRRORS_CACHE) > _MIRRORS_CACHE_MAX:
            ordered = sorted(_MIRRORS_CACHE.items(), key=lambda kv: kv[1][0])
            for k, _ in ordered[: len(_MIRRORS_CACHE) - _MIRRORS_CACHE_MAX]:
                _MIRRORS_CACHE.pop(k, None)


def _public_status_payload(job: Job) -> dict:
    """What the PWA polls for. No internal-only state."""
    return {
        "id": job.id,
        "status": job.status,
        "source_kind": job.source_kind,
        "source_label": job.source_label,
        "with_demucs": job.with_demucs,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "result": job.result,
        "error": job.error,
    }


def _result_to_payload(result: pipeline_run.MixIDResult) -> dict:
    """Compact JSON payload from a MixIDResult. No file paths leak."""
    return {
        "source_input": result.source_input,
        "tracks": [
            {
                "start_sec": round(t.start_sec, 2),
                "end_sec": round(t.end_sec, 2),
                "artist": t.artist,
                "title": t.title,
                "score": round(t.score, 3),
                "source": t.source,
                "n_segments_merged": t.n_segments_merged,
            }
            for t in result.tracks
        ],
        "unknown_segments": [
            {"start_sec": round(s, 2), "end_sec": round(e, 2)}
            for s, e in result.unknown_segments
        ],
        "timings_sec": {k: round(v, 1) for k, v in (result.timings_sec or {}).items()},
    }


def _run_pipeline_for_job(
    job_id: str,
    audio_path: Path,
    source_kind: str,
    with_demucs: bool | None = None,
) -> None:
    """Worker function — runs in thread executor."""
    job = _JOBS[job_id]
    try:
        job.status = "running"
        job.started_at = time.time()
        result = pipeline_run.run(str(audio_path), with_demucs=with_demucs)
        payload = _result_to_payload(result)
        job.result = payload
        job.status = "done"
        job.finished_at = time.time()
        # Aggregate counter (no PII)
        stats.record_run(
            duration_secs=(
                (result.timings_sec or {}).get("audio_prep", 0)
                + (result.timings_sec or {}).get("segmentation", 0)  # rough proxy
                if not payload["tracks"]
                else max((t["end_sec"] for t in payload["tracks"]), default=0.0)
            ),
            identified=len(payload["tracks"]),
            unidentified=len(payload["unknown_segments"]),
            source=source_kind if source_kind in ("file", "url") else "file",
        )
    except Exception as exc:
        log.exception("pipeline error for job %s", job_id)
        job.error = str(exc)
        job.status = "error"
        job.finished_at = time.time()
    finally:
        # Always delete the temp audio — we never retain user audio
        try:
            audio_path.unlink(missing_ok=True)
            if audio_path.parent.name.startswith("mixid_"):
                # remove the temp dir if it was created for this run
                for child in audio_path.parent.glob("*"):
                    child.unlink(missing_ok=True)
                audio_path.parent.rmdir()
        except Exception:
            log.warning("temp cleanup failed for %s", audio_path)


# ── FastAPI app ────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(title="MixID", version="0.2.0")

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        index_html = static_dir / "index.html"
        if index_html.exists():
            return HTMLResponse(index_html.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>MixID</h1><p>Static assets missing.</p>")

    @app.get("/manifest.webmanifest")
    async def manifest():
        path = static_dir / "manifest.webmanifest"
        if path.exists():
            return FileResponse(str(path), media_type="application/manifest+json")
        raise HTTPException(404)

    @app.get("/sw.js")
    async def service_worker():
        path = static_dir / "sw.js"
        if path.exists():
            return FileResponse(str(path), media_type="application/javascript")
        raise HTTPException(404)

    @app.get("/stats")
    async def get_stats() -> JSONResponse:
        s = stats.summary()
        return JSONResponse(asdict(s))

    @app.get("/load")
    async def get_load() -> JSONResponse:
        """Current queue state — drives the live status banner on the PWA.

        Returns:
          queue_depth: number of jobs ahead in the single-worker queue
                       (includes the one currently running, if any).
          state: 'free' | 'busy' | 'swamped'
          eta_minutes_for_1h_mix: rough estimate users can plan around.
        """
        active = [j for j in _JOBS.values() if j.status in ("pending", "running")]
        depth = len(active)
        if depth == 0:
            state = "free"
        elif depth <= 2:
            state = "busy"
        else:
            state = "swamped"
        # Conservative estimate: ~30 minutes per 1-hour mix in Shazam-only mode.
        # Multiply by depth + 0.5 (yours-after-the-queue).
        eta_min = int(30 * (depth + 0.5))
        return JSONResponse({
            "queue_depth": depth,
            "state": state,
            "eta_minutes_for_1h_mix": eta_min,
        })

    @app.post("/jobs")
    async def create_job(
        background: BackgroundTasks,
        file: UploadFile | None = File(default=None),
        url: str | None = Form(default=None),
        with_demucs: str | None = Form(default=None),
    ) -> JSONResponse:
        """Start a job from a file upload OR a URL paste. Returns {job_id}.

        with_demucs: omit / null → pipeline decides automatically based on
                                   how many segments come back unidentified.
                     'true'     → force Demucs on
                     'false'    → force Demucs off
        """
        if not file and not url:
            raise HTTPException(400, "Provide either a file upload or a url field.")
        if file and url:
            raise HTTPException(400, "Provide a file OR a url, not both.")

        # Resolve audio source to a local path
        if file:
            tmpdir = Path(tempfile.mkdtemp(prefix="mixid_up_"))
            safe_name = Path(file.filename or "upload.bin").name
            audio_path = tmpdir / safe_name
            content = await file.read()
            audio_path.write_bytes(content)
            source_kind = "file"
            source_label = safe_name
        else:
            if not url_input.is_supported_url(url or ""):
                raise HTTPException(
                    400,
                    f"Unsupported URL host. Try: {', '.join(url_input.SUPPORTED_HOSTS)}",
                )
            try:
                dl = url_input.download(url or "")
            except url_input.PlatformBlockedError as exc:
                raise HTTPException(
                    502,
                    f"{exc.platform} is blocking our server right now (free cloud "
                    f"hosts get rate-limited by {exc.platform}, it's not your link). "
                    f"Paste the SoundCloud, Mixcloud, or Audiomack version of this "
                    f"mix and it'll work, or upload the file directly.",
                ) from exc
            except Exception as exc:
                raise HTTPException(502, f"Couldn't fetch that link: {exc}") from exc
            audio_path = dl.path
            source_kind = "url"
            source_label = dl.title or (url or "")

        # Parse with_demucs tri-state: None / 'true' / 'false'
        wd_override: bool | None
        if with_demucs is None or with_demucs == "" or with_demucs == "auto":
            wd_override = None
        elif with_demucs.lower() == "true":
            wd_override = True
        else:
            wd_override = False

        job_id = uuid.uuid4().hex[:12]
        job = Job(
            id=job_id,
            status="pending",
            source_kind=source_kind,
            source_label=source_label,
            with_demucs=bool(wd_override) if wd_override is not None else False,
        )
        _JOBS[job_id] = job

        # Submit to single-worker executor — naturally serializes runs
        _EXECUTOR.submit(_run_pipeline_for_job, job_id, audio_path, source_kind, wd_override)

        return JSONResponse({"job_id": job_id})

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> JSONResponse:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "job_id not found")
        return JSONResponse(_public_status_payload(job))

    @app.get("/mirrors")
    async def get_mirrors(url: str = Query(..., min_length=10)) -> JSONResponse:
        """Find mirror URLs on YouTube/SoundCloud/Mixcloud/Audiomack for a
        Spotify or Apple Music link. Result is cached for an hour."""
        if not mirror_search.is_locked_platform(url):
            raise HTTPException(422, "Only Spotify or Apple Music URLs need mirror search.")
        cached = _mirror_cache_get(url)
        if cached is not None:
            return JSONResponse(cached)
        loop = asyncio.get_running_loop()
        try:
            payload = await loop.run_in_executor(None, mirror_search.find_mirrors, url)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except RuntimeError as e:
            raise HTTPException(502, str(e))
        _mirror_cache_put(url, payload)
        return JSONResponse(payload)

    # ── Spotify playlist export (optional, user-initiated OAuth) ───────────

    @app.get("/spotify/configured")
    async def spotify_configured() -> JSONResponse:
        """Lets the PWA know whether to show the 'Add to Spotify' button."""
        return JSONResponse({"configured": spotify_export.is_configured()})

    @app.post("/spotify/playlist")
    async def create_spotify_playlist(job_id: str = Query(...)) -> JSONResponse:
        """User-facing endpoint: creates a public playlist on the HOST's
        Spotify account using the host's stored refresh token. No user
        OAuth needed."""
        job = _JOBS.get(job_id)
        if job is None or not job.result:
            raise HTTPException(404, "Job not found or not finished yet.")
        if not spotify_export.is_configured():
            raise HTTPException(
                503,
                "Spotify export not configured. The host needs to run "
                "`python -m mixid.web.spotify_setup` once.",
            )
        try:
            res = spotify_export.create_playlist_on_host(
                tracks=job.result["tracks"],
                name_suffix=(job.source_label or "")[:60],
            )
        except spotify_export.NoHostConfigured as exc:
            raise HTTPException(503, str(exc)) from exc
        except Exception as exc:
            log.exception("Spotify export failed for job %s", job_id)
            raise HTTPException(502, f"Spotify export failed: {exc}") from exc
        return JSONResponse({
            "playlist_url": res.playlist_url,
            "playlist_id": res.playlist_id,
            "tracks_added": res.tracks_added,
            "tracks_unmatched": len(res.tracks_unmatched),
        })

    return app


app = create_app()


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
