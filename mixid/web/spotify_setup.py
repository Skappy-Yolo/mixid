"""One-time CLI: authorize the host's Spotify account and print a refresh token.

Run once: `python -m mixid.web.spotify_setup`
It will:
  1. Print a Spotify authorize URL — open it in your browser
  2. Log in to YOUR Spotify account (the one playlists should land in)
  3. After Spotify redirects you, paste the redirected URL back here
  4. The script extracts the refresh token and prints it
  5. Add the printed line to your .env: SPOTIFY_HOST_REFRESH_TOKEN=...

You only need to run this once per Spotify account. The refresh token
doesn't expire unless you revoke it from your Spotify Dashboard.

Before running, make sure your Spotify app on https://developer.spotify.com/dashboard
has http://localhost:8000/spotify/callback (or whatever SPOTIFY_REDIRECT_URI is)
registered under Redirect URIs.
"""
from __future__ import annotations

import sys
from urllib.parse import parse_qs, urlparse

import config
from mixid.web import spotify_export


def main() -> int:
    if not (config.SPOTIFY_CLIENT_ID and config.SPOTIFY_CLIENT_SECRET):
        print(
            "ERROR: SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET not set in .env.\n"
            "Register a dev app at https://developer.spotify.com/dashboard and "
            "set both values in your .env first.",
            file=sys.stderr,
        )
        return 1

    auth = spotify_export._auth_manager()
    authorize_url = auth.get_authorize_url()

    print()
    print("=" * 70)
    print("MixID — Spotify host setup (one time)")
    print("=" * 70)
    print()
    print("Step 1. Confirm this redirect URI is registered in your Spotify app")
    print("        dashboard (https://developer.spotify.com/dashboard):")
    print(f"        {spotify_export.redirect_uri()}")
    print()
    print("Step 2. Open this URL in your browser (the Spotify account you log")
    print("        in with is where MixID will create playlists):")
    print()
    print(f"        {authorize_url}")
    print()
    print("Step 3. After approving, Spotify redirects you to a URL that starts")
    print(f"        with {spotify_export.redirect_uri()}?code=...")
    print("        The page itself may fail to load — that's fine. Copy the FULL URL")
    print("        from your browser address bar and paste it below.")
    print()

    redirect = input("Paste the redirected URL here: ").strip()
    if not redirect:
        print("No URL pasted; aborting.", file=sys.stderr)
        return 1

    parsed = urlparse(redirect)
    code_list = parse_qs(parsed.query).get("code", [])
    if not code_list:
        print(f"ERROR: no ?code=... in {redirect!r}.", file=sys.stderr)
        return 1
    code = code_list[0]

    print("\nExchanging code for tokens…")
    try:
        token_info = auth.get_access_token(code, as_dict=True, check_cache=False)
    except Exception as exc:
        print(f"ERROR: token exchange failed: {exc}", file=sys.stderr)
        return 1

    refresh_token = token_info.get("refresh_token") if token_info else None
    if not refresh_token:
        print("ERROR: Spotify did not return a refresh token. Response:",
              token_info, file=sys.stderr)
        return 1

    print()
    print("=" * 70)
    print("SUCCESS. Add this line to your .env:")
    print()
    print(f"SPOTIFY_HOST_REFRESH_TOKEN={refresh_token}")
    print()
    print("=" * 70)
    print("After saving .env, restart `python -m mixid --serve` and the 'Add to")
    print("Spotify' button on tracklists will work for every user.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
