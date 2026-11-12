# Recorded demo source

`revision036-source.tar.gz` is the exact source of the recorded scripted demo. SHA-256: `66c906d90897aefe31fa6f1290fa4e36721c829eb12e257f8f759f22a92c62ec`.

The current `runtime/` adds the particle-capable Inspect Robots interface and uses measured pin contact to detect wafer return. The primary demo score recomputes these particle conditions from all 234,601 saved physics samples and combines them with the complete independent handling/geometry audit of that same trace. This is an offline rescore of the recorded run, not a new policy rollout.

`evidence/primary-score-demo.json` binds that result to the trace and scorer hashes. `evidence/video-replay-manifest.json` binds the video's 20 Hz measured states to the full trace. Large raw traces and videos are not included in Git.
