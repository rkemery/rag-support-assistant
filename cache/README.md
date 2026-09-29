# Replay cache

Model responses from live runs, one JSON file per request under `<first two hex chars>/<sha256>.json`, keyed by the sha256 of the canonical request (the harness `CachedClient` format). Each file holds the request and the response with its token usage, so a cache can be read and diffed by hand.

It holds every call from the live run (about 4,300 files, 59 MB) and is committed, so `make eval-replay` (or `eval run --replay`) reproduces the live run offline, failing on any request it has not seen.
