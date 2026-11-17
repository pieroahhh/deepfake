# Unchanged-modality stream reuse

This execution optimization was applied before the remaining AVTENet study
started, while FK robustness continued. Completed outputs and the active FK
worker are unchanged. No clip, window, threshold, checkpoint or epoch is removed.

The AVTENet runner now checks the original and derivative modality streams before
reusing an expert vector. Identity covers compressed packet bytes, codec config,
container origin, stream start and rational packet PTS/DTS/durations. A mismatch
falls back to the original raw preparation/inference path. The original feature
store's completion/checksums and source/window bindings are still verified.

For the 475 derivatives, a CPU-only population scan verified:

| Condition | Unchanged modality | Verified cases |
|---|---|---:|
| Video compression | Audio / AST | 200/200 |
| Audio compression and real audio delay | Video / ViViT | 275/275 |

Together with existing reuse of the 100 clean cases, this avoids raw inference
on 300/575 AST cases and 375/575 ViViT cases. AV-HuBERT still computes the changed
audiovisual inputs. Fusion still uses the locked checkpoint and threshold.

The scan took 3.82 seconds on this machine. This is verification overhead, not
a measurement of total GPU runtime saved. Case counts cannot be treated as a
percentage reduction in the entire queue's wall time.

Profiling deliberately runs raw inference on every timed repeat. Reused study
samples have null timing fields and explicit execution/stream provenance, and
cannot enter runtime summaries as measured zero-second predictions.

Evidence: `runs/remaining-supplement-002/stream-reuse-population-audit.json` and
`stream-reuse-checks.json`. The test checks identical stream matching, changed
video rejection, missing audio and invalid stream selection. Each actual run
rechecks stream identity, so the audit is not a substitute for runtime checks.
