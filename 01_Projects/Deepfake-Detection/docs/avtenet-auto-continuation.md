# Automatic AVTENet continuation

The user authorized automatic continuation on 16 November 2026. The original
AV-HuBERT training remains running. A separate background controller waits for its
completed feature cache and release of the workflow lock, then performs:

1. Verify selected AST, ViViT and AV-HuBERT checkpoints, training identities,
   feature hashes, SQLite integrity, finite vectors and source/window alignment.
2. Merge aligned train/validation windows and train fusion for one epoch (seed 7).
3. Evaluate validation and export all selected checkpoints/provenance to
   `avh-a/return-c-validation`.
4. Freeze all four checkpoint hashes before any held-out inference.
5. Extract audio, visual and audiovisual features for the same frozen held-out
   1,000 clips used for FK and Machine B. These are unused official validation
   clips, not the official hidden-label test set.
6. Evaluate frozen fusion at threshold 0.5. Retain every clip in the output and
   coverage denominator, including clips without a complete three-expert window.

Results are written under `code/runs/auto-next-001/`:

- `status.json`: overall automatic continuation state.
- `expert-verification.json`: selected checkpoints and verified feature counts.
- `test-plan.json`: immutable inputs and selected weights for held-out inference.
- `heldout/evaluate/report.json` and `predictions.jsonl`: final test results.
- `summary.json`: validation and held-out metrics plus output paths.
- Separate stage logs and the background controller launcher record.

## Watch

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\teeku\Desktop\fk\avh-a\watch.ps1'
```

The watcher now shows the current training stage and the automatic continuation
state. `WAITING / avhubert_training_and_features` is expected until AV-HuBERT
finishes. The controller polls every 30 seconds and does not occupy the GPU while
waiting. CUDA stages acquire the same cross-checkout GPU queue.

## Pause and resume

Use `pause.ps1` as documented in HANDOFF.md. All stages honor the pause marker
at their next safe check. Wait for the active stage to stop before closing Docker.
The continuation controller stops on upstream pause/failure or any verification
error; it does not silently bypass errors or restart the computer.

After an AV-HuBERT pause/failure, resolve the error and run `start.ps1` to resume
AV-HuBERT, then `continue-auto.ps1` to resume its waiting controller. If AV-HuBERT
and its feature cache are already complete, clear only the pause marker and run
the continuation launcher:

```powershell
Remove-Item -LiteralPath 'C:\Users\teeku\Desktop\fk\avh-a\code\runs\worker-c\avtenet\pause.request' -ErrorAction SilentlyContinue
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\teeku\Desktop\fk\avh-a\continue-auto.ps1'
```

Fusion uses periodic training checkpoints. Interrupted merge/evaluation folders
are preserved and regenerated from frozen inputs. Held-out extraction commits
successful rows periodically and reuses them on restart; incomplete work since
the last commit is repeated. Preparation failures can be retried and are never
assigned fabricated predictions. Completed caches and outputs must match their
recorded hashes before they are reused. No Windows startup task is installed;
after a reboot, open Docker and relaunch the appropriate scripts.

## Prepared and tested

Completed C expert artifacts were imported with SHA-256 checks, preserving the
running local AV-HuBERT code. All 1,000 held-out media files were verified and
copied under the local dataset root; the training index remains the frozen 3,500
rows. ID, source-family and content-hash disjointness is checked before testing.

Targeted tests cover frozen inference, interruption/recovery, all-1,000 reporting
with unassessed clips, completed-output reuse, changed-checkpoint rejection, and
safe artifact import. Full real fusion/test results remain pending until the
upstream AV-HuBERT job completes.
