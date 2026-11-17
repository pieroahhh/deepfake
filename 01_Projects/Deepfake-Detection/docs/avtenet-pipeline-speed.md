# Remaining-pipeline optimization — 16 November 2026

The active AV-HuBERT feature extractor continued running throughout this update.
Only its idle downstream controller was briefly stopped and relaunched. Existing
training, labels, sample order, precision, batch size, and decision thresholds were
preserved. These changes target preparation and data movement, not model quality.

## Enabled changes

1. **ViViT input packing:** run the same local image processor with NumPy output,
   then construct a tensor directly from its contiguous array. This avoids the
   expensive nested-list-to-tensor conversion. Cached input transformation also
   runs ahead on one CPU worker, with a bounded depth of four.
2. **AST audio preparation:** decode/resample and transform upcoming audio on a
   CPU worker while the main thread runs GPU inference. This includes missing
   raw cache entries, not just reads of already prepared entries.
3. **CPU prewarm:** a separate two-worker process is preparing all 1,000 held-out
   clips' raw audio while the current GPU job finishes. Its cache binds the exact
   manifest, source media hashes, PyAV/libav versions and media adapter code.
   It is reused by host AST only; different AV-HuBERT decoder versions do not
   silently share it. The controller waits for prewarm completion before AST
   extraction so two writers cannot race on the same cache entries.
4. **Lossless raw cache:** new held-out entries use uncompressed NPZ instead of
   compressed NPZ. Original arrays, checksums, corruption checks and atomic
   payload publication remain. Existing entries are readable in either format.
5. **Evaluation:** retrieve each expert's features once per clip, then align
   windows in memory. Model inference still uses the original window order and
   batch size. Hash, source, feature-shape and finiteness checks are retained.

The existing single-GPU queue, one-thread OpenBLAS setting, decoder/detection
reuse, exact feature resume check, checkpoints and failure accounting remain.
File transfer/preparation is already local on SSD and completed; copying the
whole dataset again would not speed up this experiment.

## Measurements and tests

`code/runs/auto-next-001/preprocessing-benchmark.json` records bounded CPU tests:

- The actual local ViViT image processor on 32 synthetic RGB frames took
  **0.631 seconds -> 0.053 seconds**; output tensors were bit-exact.
- On a synthetic high-entropy visual window, NPZ write/materialization took
  **0.182 seconds compressed -> 0.021 seconds uncompressed**. Compression ratios
  and performance on real faces vary; this is not a whole-pipeline measurement.
- Three real held-out clips / 18 usable audio windows produced bit-exact AST
  inputs. CPU-only elapsed time was **0.475 -> 0.457 seconds**; this experiment
  does not measure the additional benefit of overlap with GPU inference.
- Three targeted tests cover cold-cache AST preparation/failure order/RNG,
  lossless cache hit/miss and corruption handling, and aligned SQL reads.
- Two workflow tests pass on the host and the pinned Torch 1.13 container:
  frozen checkpoint validation, interrupted extraction recovery, output reuse,
  unassessed clip accounting, and verified artifact import.

No end-to-end speed multiplier or earlier completion time is guaranteed from
these component timings. Observe the actual held-out stage logs for updated ETA.
Source backups are in `source-before-heldout-speed`; a hash inventory is kept in
`heldout-speed-source.json`. Source/test changes are also copied to the main repo.

## Disk and progress

At preparation time C had approximately **119.7 GiB free**. If all 8,613 held-out
windows are cached, the raw visual/audio/lip arrays require roughly **44 GiB**,
plus metadata, features and small overheads. There is an 8 GiB free-space guard
before each new raw-cache write. Deleting cache while jobs are running is unsafe.

Use the existing `watch.ps1`; it now displays `CPU audio preparation` as well as
the GPU stage and automatic queue. Detailed audio prewarm state and logs:

`code/runs/auto-next-001/audio-prewarm/status.json`

`code/runs/auto-next-001/audio-prewarm.stdout.log`

The CPU prewarm honors `pause.request`; on failure its controller stops before
AST extraction and reports the problem. To rerun the prewarm after resolving a
failure, ensure no earlier prewarm worker is running and run from the code folder:

```powershell
& 'C:\Users\teeku\Desktop\fk\Deepfake-Detection\.venv-gpu\Scripts\python.exe' 'C:\Users\teeku\Desktop\fk\avh-a\code\scripts\prewarm_heldout_audio.py'
```

Completed entries are reused. Then relaunch `continue-auto.ps1` if necessary.
The final frozen test still reports all 1,000 clips, including unassessed ones.
