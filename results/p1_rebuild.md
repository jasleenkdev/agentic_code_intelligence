# P1 — full vs incremental index rebuilds

Model `gte-modernbert`, CPU only (Apple M1 Pro). Fresh caches for this run (the first
build of each library is a true full build). Incremental = same command on the next
version:
files are re-parsed only if their git blob changed, snippets re-encoded only if their
text changed (content-hash embedding cache).

| Library | Version | Build | Snippets | Files parsed / reused | Encoded / reused | Chunk s | Encode s | Total s |
|---|---|---|---|---|---|---|---|---|
| gladys | v5.0.4 | full | 6424 | 2152 / 0 | 6422 / 2 | 2.0 | 468.1 | 470.2 |
| gladys | v5.1.0 | incremental | 6642 | 146 / 2041 | 394 / 6248 | 0.52 | 50.0 | 50.8 |
| gladys | v5.1.1 | incremental | 6642 | 1 / 2186 | 1 / 6641 | 0.13 | 0.4 | 0.8 |
| apps-derived | v1 | full | 2000 | — | 2000 / 0 | 0.0 | 154.2 | 154.3 |
| apps-derived | v2 | incremental | 2000 | — | 176 / 1824 | 0.0 | 15.9 | 16.0 |
| apps-derived | v3 | incremental | 2000 | — | 23 / 1977 | 0.0 | 3.7 | 3.8 |
