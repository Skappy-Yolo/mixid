"""MixID web tier: PWA-style HTTP interface to the pipeline.

The CLI/pipeline is the engine; this package wraps it as a single-page
web app (FastAPI backend + PWA frontend) that anyone can use from a phone
without logging in. Hosted from the user's laptop via Cloudflare Tunnel.
"""
