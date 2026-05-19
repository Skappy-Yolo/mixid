# MixID — Farcaster thread (8 casts)

Format: each block is one cast, ~280 chars or so. Casts numbered for your sanity but don't include numbers in the actual posts.

---

## 1/

came back from a belgian rave couldn't stop thinking about the set. tried shazam on my phone recording, identified one snippet, gave up on the rest. tried 1001tracklists, dj not listed. so i spent 3 days building a tool that takes the recording → tracklist.

→ github.com/Skappy-Yolo/mixid

## 2/

constraints: $0/month, no login, runs on a laptop. anyone uploads a mix → timestamped tracklist comes back. file upload or paste a youtube/soundcloud/mixcloud/audiomack url.

[image: screenshot of the PWA on phone]

## 3/

first version was a lie. it "identified" 43 tracks in a 23min rave recording. the tracklist was disney pop, k-pop, irish country songs, gospel. confidently hallucinated.

root cause: i'd set my score floor at 0.55. chromaprint's random-noise baseline is 0.50. i'd been accepting noise as matches.

## 4/

real fix was admitting silence. raised floor to 0.80. required both max AND median across pitch variants. added 5-word minimum on transcripts. re-ran: 0 hits. zero is the right answer for that pipeline on that audio. better to say "i don't know" than to lie.

## 5/

the actual unlock was an old library that came back from the dead. shazamio (unofficial python wrapper for shazam) broke in 2024. someone fixed it sometime in 2025. tried it 3 days into the build, identified niche dutch rap on first try. wired in as primary matcher.

## 6/

biggest single accuracy lever in the system isn't a model or an ensemble. it's a ±6% pitch sweep before fingerprinting. djs pitch-shift tracks to beatmatch, standard chromaprint silently misses every shifted one. fingerprint 7 variants per sample, take the best. +15-20pp recall, pure cpu.

## 7/

llm re-rank uses claude code cli as subprocess instead of an api. uses my existing claude code subscription, no key, no quota. burned 3 gemini keys debugging "prepayment credits depleted" errors on brand-new free-tier keys before realizing. sometimes the workaround is the answer.

## 8/

real numbers on a 58-min rave recording, phone in pocket, friends chatting:
— 30 unique tracks (fast pass, 30 min runtime)
— 34 with deep scan (demucs vocal-stem retry, 2h 45m runtime)

dutch rap, french afro-trap, classic hip-hop, dancehall, afrobeats.

mit licensed. try it free at [paste your pwa url]. fork the repo, build your own.

---

## Notes for posting:
- Pair the cast with a real screenshot of the PWA / tracklist output
- The Substack link at the end optional, depends on if it's published yet
- If you want to keep it shorter, casts 6+7 can be cut without losing the story
- Tag dev-FC accounts you respect for visibility (don't @ anyone in the post itself, do it in a reply)
