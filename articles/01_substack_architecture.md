# How I Built a Shazam for Hour-Long DJ Mixes in 3 Days

*And what it taught me about engineering with no budget, no servers, and no patience for fake metrics.*

---

I went to a rave in Belgium last weekend. Some Dutch DJ, somewhere in the city. The set was unreal — French Afro-trap into Sean Paul into a Kendrick mashup into Burna Boy. By Sunday afternoon I could remember exactly half a song, no titles, no artists. I tried Shazam from my phone recording: it identified one 30-second snippet, ignored the other 22 minutes. I tried searching the DJ's name on 1001tracklists: not listed. I asked friends: "yeah I remember the drop, no idea what it was."

This is the same problem millions of people have. Shazam is built for the radio era — give it a clean 10 seconds, it identifies one song. Show it an hour-long mix recorded by a friend through a phone in a crowded room and it gives up. There's no free tool that takes "I have a recording from a rave, what was playing?" and returns a tracklist.

So I built one.

The repo is at [github.com/Skappy-Yolo/mixid](https://github.com/Skappy-Yolo/mixid). The live demo is at [paste-your-cloudflare-url-here]. This is the engineering story — the architecture decisions, the false positives that almost shipped, and the constraint that turned out to be the unlock.

---

## The brief

- **Input:** a recording of a DJ set (your phone, a YouTube rip, a Mixcloud URL, anything). Could be 5 minutes, could be 3 hours.
- **Output:** a timestamped tracklist. "00:09 Artist - Track Title."
- **Constraints I gave myself:**
  - $0/month. No paid APIs.
  - No login. Anyone uses it instantly.
  - Works on a normal laptop, not a GPU cluster.
  - Honest. If we can't identify a section, the output says "unidentified" — never invents a song.

## The first version was a lie

V1 ran in about a day. URL shortcut → audio prep → segmentation → Chromaprint fingerprint → AcoustID lookup → tracklist. I tested it on a clean studio track I knew: nailed it, 97% confidence.

Then I tested it on the actual 23-minute rave recording. It identified 43 tracks.

Felt good for about 90 seconds, until I read the tracklist.

```
00:09  BABYMONSTER - I LIKE IT  [score=0.61]
01:30  Big Time Rush - Elevate  [score=0.59]
02:30  None Like Joshua - Super Mario Dubstep Cypher
03:00  La Mano 1.9 - I'm Sorry
14:25  Daniel O'Donnell - I'm Going to Be a Country Boy Again
22:54  Up with People - Where The Roads Come Together
```

This was supposed to be a Belgian rave. The output identified it as Disney pop, K-pop, Irish country, and gospel. I'd built a thing that confidently lies.

Root cause: I'd set the score floor at 0.55. Chromaprint's audio hash has a *random baseline of ~0.50* — two random 32-bit hashes differ in ~16 of 32 bits on average. Anything 0.55-0.65 isn't a match, it's noise. Combined with Whisper transcribing chit-chat from my friends ("yeah man, let go, I'm going") which fuzzy-searched fine against thousands of unrelated songs, I'd created a hallucination machine.

The fix was three lines: raise floor to 0.80, require both max AND median across pitch variants, reject transcripts under 5 words. Re-ran: 0 hits.

**Zero is the right answer for that pipeline on that audio.** Better to admit you don't know than to confidently say "Up With People." I shipped the fix, accepted the silence, and started thinking about what would actually work.

## The real unlock was an old library that came back from the dead

I had a note in my notes file: "*shazamio is broken — Apple changed the protocol in 2024.*" Saw that during the build, assumed it stayed broken, didn't try it.

Three days into the project, frustrated by AcoustID's coverage gaps (it just doesn't have most electronic / Afrobeats / DJ-edit tracks), I tried `shazamio` anyway. It worked. Someone had fixed the protocol some time in 2025. The same library that returned nothing on a track in March now identified obscure French Afro-trap on the first try.

Wired it in as the primary remote matcher, ran the rave audio again:

```
01:00  Chivv & Diquenza - Ewa Ewa
03:00  MHD - Afro Trap, Pt. 7 (La puissance) [Major Lazer Remix]
05:59  TRIANGLE DES BERMUDES - Charger
07:27  JOHN PATTON - Band For Band
11:26  SBMG & Diquenza - Pull Up Game Strong
13:37  Sean Paul - Temperature
22:54  Akon Feat. Eminem - Smack That (Crispy Remix)
...
```

This actually looked like a Belgian rave. Dutch rap, French Afro-trap, the inevitable Sean Paul. The pipeline was honest now: the lineup matched what humans played, the silences matched where chatter drowned the music.

## The architecture (no shortcuts hidden)

```
┌──────────── Input ────────────┐
│ Audio file OR streaming URL    │
└─────────────┬─────────────────┘
              ▼
┌──────────────────────────────────────┐
│ URL shortcut: mixesdb wiki or YT     │
│ description for known tracklists     │
└─────────────┬────────────────────────┘
              ▼ miss
┌──────────────────────────────────────┐
│ Audio prep (ffmpeg/librosa):         │
│  mono 22050 Hz, loudness norm,        │
│  highpass 80 Hz                       │
└─────────────┬────────────────────────┘
              ▼
┌──────────────────────────────────────┐
│ Hybrid segmentation:                  │
│  novelty curve + beat-grid phrase     │
│  + forced 90s gap cap                 │
└─────────────┬────────────────────────┘
              ▼
┌──────────────────────────────────────┐
│ Per segment: top-3 best-sample picks │
│ → Chromaprint fingerprint sweep      │
│   (±6% pitch variants — DJs pitch     │
│    tracks to beatmatch, naive fp      │
│    misses every one)                  │
└─────────────┬────────────────────────┘
              ▼
   Local library (opt-in cache)
   → audio recognition libraries (free)
   → AcoustID + MusicBrainz (≥0.80 floor)
   → reactive lookup: Whisper-tiny
     transcribes → iTunes/Deezer search
     → preview download → fp-verify
              ▼
┌──────────────────────────────────────┐
│ Auto Deep Scan trigger:               │
│  if >50% segments still unidentified, │
│  Demucs htdemucs_ft separates stems,  │
│  retry recognition on no-vocals stem  │
│  (rescues hits buried under crowd     │
│  chatter)                             │
└─────────────┬────────────────────────┘
              ▼
   LLM re-rank (Claude Code CLI as
   subprocess — uses my Claude
   subscription, no API key burn)
              ▼
   Gap-fill smoother (rescue unknowns
   sandwiched by same-track neighbors)
              ▼
   Consensus collapse + output: JSON,
   M3U, plain text + "Add to Spotify"
```

Some pieces worth zooming in on:

### The pitch-shift sweep is the biggest single lever

Chromaprint is brittle to the ±3–6% pitch shift DJs apply for beatmatching. A naive fingerprint silently misses every track the DJ touched. I fingerprint 7 variants per sample (0%, ±2%, ±4%, ±6%) and take the best. Pure CPU, sub-second per sample, **+15–20pp recall**.

### Reactive lookup is harder than it sounds

When the pre-built catalogs miss, I transcribe a vocal snippet with Whisper-tiny, search iTunes + Deezer with the resulting phrase, download each candidate's 30-second preview, fingerprint it, compare to the query. This works in theory. In practice it's where the hallucinations almost killed the project. The fix: 0.80 score floor + median check across pitch variants + 5-word minimum on transcripts + a stoplist of generic phrases ("I'm going", "let go", "thank you").

### LLM via Claude Code subprocess > LLM via API

I tried Gemini's free tier. Burned through three API keys in debugging because Google's response to `system_instruction + generationConfig` is "prepayment credits depleted" even on a brand-new free-tier key. The fix: shell out to the `claude` CLI from a Python subprocess. Uses my existing Claude Code subscription. No API key, no quota burn, no rate-limit roulette. The re-ranker only fires on ambiguous segments anyway (most segments have one or zero candidates), so the cost is bounded.

### Deep Scan auto-decides

I had this as a checkbox in the UI for a day. Removed it. Users don't know what Demucs is. They click "Generate tracklist." If after the fast pass more than half the segments are still unknown, Demucs runs automatically in the background. Browser notification fires when it's done so they can come back.

## Real numbers from a 58-minute Belgian rave

I ran the full pipeline on the actual recording. Results, no cherry-picking:

| Metric | Value |
|---|---|
| Mix duration | 58 minutes |
| Recording | phone in pocket, friends chatting |
| Fast pass (no Deep Scan) | 30 unique tracks, 30 minutes runtime |
| Full pass (auto-triggered Deep Scan) | 34 unique tracks, 2h 45min runtime |
| AcoustID false positives blocked | 2 (Anthony Robbins audiobook, violin concerto) |

The tracklist is real and the right genre. Dutch rap, French Afro-trap, classic hip-hop, dancehall, Afrobeats. Fetty Wap, Eminem, Sean Paul, Drake, Aya Nakamura, Burna Boy, Kendrick, Kevin Lyttle as the closer.

The full text [is at this gist] — I cross-referenced four tracks with the DJ's Instagram setlist post the next day. All four matched.

## What doesn't work, honestly

**Anything I couldn't engineer past after applying hacker thinking:**

1. **No free, official, server-side Shazam-quality API exists.** Apple's deliberate architecture after acquiring Shazam in 2018. ShazamKit (the official one) is iOS/macOS-only — it cannot run on a Linux server. Workarounds either use reverse-engineered clients (fine for personal local use, fragile and TOS-borderline for public hosting) or paid services.
2. **Spotify's catalog cannot be fingerprint-matched without OAuth.** Preview URLs deprecated in Nov 2024. `librespot` uses Premium creds and breaks DRM — not viable for public deployment. Web Playback SDK requires user login, which kills the "no login" goal.
3. **Phone-recorded crowd-noise audio has an intrinsic SNR floor.** Information-theoretic, not algorithmic. Recording technique fixes it; software cleverness doesn't.

The right answer is to acknowledge these and design around them, not pretend they're not there.

## Ship from where you are

The whole thing runs on my laptop. There's no cloud. Public access is exposed via `cloudflared tunnel --url http://localhost:8000` — free, no signup needed. When my laptop sleeps, the demo sleeps. When my laptop is upgraded, the demo gets faster.

This is wildly under-engineered for a "production system." It's perfect for a portfolio piece. Anyone reading this can fork the repo, run it on their own machine in 15 minutes, and shape it for whatever they actually need.

The full source is MIT-licensed: [github.com/Skappy-Yolo/mixid](https://github.com/Skappy-Yolo/mixid). The live demo at [insert PWA URL] is hosted on my laptop until you read this — try a 5-minute mix recording first; the longer ones take ~30 min and queue if someone else is ahead of you.

If MixID helps you remember a song from a rave you forgot, a coffee on Ko-fi or a few cents to one of the crypto addresses keeps the laptop on. Truly optional. The whole point was building something useful for free.

---

*Built in 3 days with Claude Code as a pair-programming partner. 35+ commits, 60 passing tests, 1 PWA, 0 paid services. Repo public at [github.com/Skappy-Yolo/mixid](https://github.com/Skappy-Yolo/mixid).*
