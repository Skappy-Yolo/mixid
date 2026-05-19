# What building MixID taught me about engineering with hard constraints

I spent a weekend building a "Shazam for hour-long DJ mixes" because no free tool does that, and I'd just come from a rave I couldn't stop thinking about.

The constraints I gave myself were unreasonable:
- **$0/month.** No paid APIs.
- **No login.** Anyone uses it instantly.
- **Runs on a laptop.** Not a GPU cluster.
- **Honest.** If we can't identify a segment, the output says "unidentified" — never invents a song.

Three days, 35 commits, one PWA, zero paid services later, it works. On a 58-minute phone recording of a Belgian rave with my friends chatting in the background, the tool identified 34 unique tracks — Dutch rap, French Afro-trap, classic hip-hop, dancehall, Afrobeats. Real lineup, right genre, real timestamps.

Three things I'd carry into any future build under tight constraints:

**1. Default to silence over hallucination.**
My first version "identified" 43 tracks from the rave — Disney pop, K-pop, Irish country songs, gospel. The pipeline was confidently lying because I'd set my score floor at 0.55, which is the *random-noise baseline* for Chromaprint hashes. Better to ship zero hits than 43 wrong ones. Raised the floor, accepted the silence, started designing for honesty rather than apparent coverage.

**2. Question every "paid" assumption.**
I needed an LLM for re-ranking ambiguous segments. Google's free Gemini tier burned through three API keys in debugging (their response to certain JSON fields is "prepayment credits depleted" even on brand-new free-tier keys). I almost gave up and added it to the "things to pay for" list. Then I realized I could shell out to the `claude` CLI from a Python subprocess — uses my existing Claude Code subscription, no API key, no quota burn. The LLM problem dissolved.

**3. Find the unlock that doesn't look like an unlock.**
The biggest accuracy lift in the entire system wasn't a model upgrade or a clever ensemble. It was applying a ±6% pitch sweep before fingerprinting, because DJs pitch-shift tracks for beatmatching and standard Chromaprint silently misses every shifted track. Pure CPU. Sub-second per sample. +15-20 percentage points of recall. Sometimes the leverage is in a place nobody talks about.

What I shipped:
- **PWA** (single page, no login, file upload or URL paste): demo URL in comments
- **Source code** (Python, MIT-licensed, 60 passing tests): https://github.com/Skappy-Yolo/mixid
- **Architecture write-up** (full engineering story, every shortcut named): https://[your-substack].substack.com/p/mixid

Also: I built this with Claude Code as a pair-programming partner — not "Claude wrote it for me" but genuine partnership where I drove architecture, it implemented + caught mistakes, and adversarial review agents stress-tested every claim of mine that turned out to be sandbagging.

Open to talking about audio engineering, music-information-retrieval, or what shipping under constraints looks like in practice. The repo's MIT-licensed — fork it for anything you want.

#audioengineering #musicinformationretrieval #buildinpublic #pwa #claudecode #portfolio
