# Machine B: AVFakeNet recovery handoff

**Audience:** the Machine B operator and coding agent.  
**Date:** 16 November 2026.  
**Goal:** recover meaningful visual/audio branch learning, using training and validation data only, while preserving every completed experiment.

This is an implementation and execution handoff. **A recovery trainer with the safeguards below has not been implemented yet.** The preflight commands in this document exist now. Do not assume that copying this Markdown file changes the training code.

## 1. What Machine A confirmed

Machine A verified `return-b-diagnostics-001`, loaded both selected checkpoints strictly, and checked their identities, selected epochs and file checksums. They match the original validation and held-out reports:

| Branch | Selected epoch | Selected checkpoint SHA-256 |
|---|---:|---|
| Visual | 1 | `fab5414c7401b9bd7a6ffae7737e725f76e5ea90c27d6f358bd912ebba323abe` |
| Audio | 3 | `0ae916a31dca1719c609c9f815a62f319e7f6fcc313be8b64339c89529f44aa7` |

Confirmed observations:

- Visual validation frame accuracy is 97.73%, but the first ReLU in the output classifier, `output_block.1`, outputs **all zeros** on four diagnostic synthetic images and four real validation face crops. Those real crops include two genuine frames and two frames inside annotated visual edits. All four receive the identical fake probability **0.0670785159**.
- Visual logs record zero first-batch gradient norms in the input block, patch embedding, attention stages and merging layers in epochs 2 and 3. This is evidence of a dead feature-to-classifier path on tested inputs, not evidence of useful 97.73% detection performance.
- Audio validation accuracy is 47.78%, 47.78%, then 52.22%. On 16 real validation audio clips, eight with genuine audio and eight with fake audio, the selected checkpoint returns **0.5090539455 fake probability for every clip**. All 16 reproduce the original audio hard vote.
- Both checkpoints contain trained parameter changes and finite tensors. They are not unchanged random initializations. Their exact initiating training failure has not yet been isolated.
- The validation fused result already predicts fake on all 217 assessed clips. The held-out result predicts fake on all 417 assessed clips because audio always votes fake and fusion uses hard OR.
- Whole-video preprocessing assessed only 417/1,000 held-out clips. Major first-failure reasons are crop/landmark failure (370), source-frame gaps (133), and incomplete audio coverage (39). These counts are separate from the model-learning failure.

The diagnosis on Machine A is at:

```text
Deepfake-Detection/runs/avfakenet-checkpoint-audit-001/DIAGNOSIS.md
Deepfake-Detection/runs/avfakenet-checkpoint-audit-001/report.json
Deepfake-Detection/runs/avfakenet-checkpoint-audit-001/real-visual-probes.json
Deepfake-Detection/runs/avfakenet-diagnostics-002/DIAGNOSIS.md
```

These are Machine A result paths; they are not guaranteed to exist on Machine B. The findings above are enough to begin the recovery implementation.

## 2. Preserve the completed work

Do not overwrite, delete, resume training from, or regenerate the original failed experiment. Preserve:

```text
runs/worker-b/avfakenet/
return-b-to-machine-a/                 # if present
return-b-heldout-1000-to-machine-a/    # if present
return-b-diagnostics-001/              # if present
```

The old launcher uses a nested model directory:

```text
runs/worker-b/avfakenet/               # workflow output
  avfakenet/                         # actual branch outputs
    visual/attempt-001/
    audio/attempt-001/
    evaluation/
```

Create a new namespace such as `runs/avfakenet-recovery-001`. Keep all checkpoints, profiles, caches, progress and reports for recovery under it. A retry must use a new attempt directory or a verified recovery checkpoint with matching identity. Never resume from the old collapsed `best.pt` or `last.pt`.

Keep the original media, population manifest, labels, source hashes and official annotations unchanged. Do not modify FK, AVTENet or the controls. Do not silently replace the paper-style majority vote or hard OR.

## 3. Preflight commands that work now

Run this as one block in PowerShell. Change **only the two path settings** to match the laptop. The example uses the previously recorded Machine B layout. For a portable bundle, the dataset may instead be the sibling `..\dataset` folder.

```powershell
& {
    $ErrorActionPreference = 'Stop'

    $RepoRoot = 'D:\github\Deepfake-Detection'
    $DatasetRoot = 'D:\github\Deepfake-Detection\dataset'

    Set-Location -LiteralPath $RepoRoot
    $DatasetRoot = (Resolve-Path -LiteralPath $DatasetRoot).Path
    $PythonPath = (Resolve-Path -LiteralPath '.\.venv-gpu\Scripts\python.exe').Path
    $ManifestPath = (Resolve-Path -LiteralPath '.\runs\two-day-001\population\manifest.json').Path

    $ExpectedPopulation = '3c0955b7daaeed0d22b0efe68d06ee78146a8f43c88a5c3df28e1db3e13f5fea'
    if ((Get-FileHash -LiteralPath $ManifestPath -Algorithm SHA256).Hash -ne $ExpectedPopulation) {
        throw 'Wrong training/validation population. Do not prepare a replacement subset.'
    }

    $Rows = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    if ($Rows.Count -ne 3500) { throw "Expected 3500 clips; found $($Rows.Count)" }
    if (@($Rows | Where-Object split -eq 'train').Count -ne 3000) { throw 'Expected 3000 training clips' }
    if (@($Rows | Where-Object split -eq 'validation').Count -ne 500) { throw 'Expected 500 validation clips' }
    if (@($Rows | Where-Object split -eq 'test').Count -ne 0) { throw 'Recovery population must not include held-out test clips' }

    foreach ($Row in $Rows) {
        $ClipPath = Join-Path $DatasetRoot $Row.path
        if ((Get-FileHash -LiteralPath $ClipPath -Algorithm SHA256).Hash -ne $Row.source_sha256) {
            throw "Source checksum mismatch: $ClipPath"
        }
    }

    & $PythonPath scripts/check_worker_cuda.py
    if ($LASTEXITCODE -ne 0) { throw 'CUDA check failed' }
    Write-Host 'Verified CUDA and all 3500 train/validation source clips'
}
```

Do not re-run `machine_b.ps1 -Run` or `run_two_day.py` as the recovery solution. Those commands retain the old streaming order and training policy, and may simply reuse already completed stages.

Do not use `--limit-per-split` as a substitute for a balanced diagnostic sample: it is not a class-balanced sampler. Existing `--max-steps` runs are marked integration-only; they must not be reported as final benchmark training.

## 4. Implementation work before recovery training

The coding agent on B should make these changes in a separate recovery entry point, keeping the original baseline behavior available.

Relevant existing files:

```text
baselines/avfakenet/pipeline.py        # Loader, train, evaluate
baselines/avfakenet/model.py           # DSTNet and classification_loss
baselines/avfakenet/prepared_cache.py  # prepared media and identity checks
baselines/common/training.py          # optimizer, selection, checkpoints
baselines/common/population.py        # modality/temporal targets and source checks
scripts/run_two_day.py                # old workflow; do not use unchanged
```

### A. Deterministic ordering and input provenance

- Shuffle training examples deterministically using the run seed and epoch. The current loader orders clips by ID and streams consecutive frames from each clip.
- For the tiny sanity set, choose actual post-preparation samples with known labels. Use training rows only, multiple source families and a cap per clip so one long video does not dominate.
- Preserve source timestamps. Visual frame labels must come from the frame's annotated source time, not the clip-level fake label. Do not label every frame of a partially modified clip as fake.
- Save sample IDs, source hashes, timestamps, label counts and the selection seed before optimization.
- A class-balanced sanity set changes the sampling distribution; record that explicitly. It is a diagnostic, not the final experimental population.

### B. Keep the architecture/loss fixed for the first recovery attempt

- Reinitialize each branch from the recorded seed, initially 7.
- Keep DSTNet architecture, image sizes, normalization, AdamW, BCE convention and label smoothing 0.1 unchanged.
- Start with learning rate **1e-4**, instead of the failed run's 1e-3. Record this as a recovery setting, not a hyperparameter attributed to the original paper.
- Batch size 8 and four CPU threads are reasonable starting settings for the 16 GB RAM / 8 GB VRAM laptop. Run one GPU training job at a time.
- Do not introduce AMP, a new activation, weighted loss, warmup and gradient clipping simultaneously. If stability still fails, test a controlled change on training/validation and record it separately. The audit does not prove that learning rate alone caused the collapse.

### C. Health telemetry and failure detection

At minimum, record per branch:

- Actual prepared sample counts by class, skipped cases and their reasons.
- Training and validation loss, confusion matrix, ordinary accuracy and balanced accuracy.
- Raw two-class logits, fake-probability spread and class proportions on a fixed small diagnostic set.
- Gradient norms for input block, patch embedding, each stage/merge and the classifier, before any clipping.
- Zero-activation fraction at `output_block.1` and `output_block.4`.
- Activation magnitude and across-example variation at stage 3, the final merge and classifier input.

The old visual model had large stage-3 activations, almost invariant merged features, and a completely zero first classifier ReLU on tested inputs. The recovery monitor must make this visible during training.

Use a conservative stop guard for repeated nonfinite loss/gradients, an entirely zero first classifier ReLU together with absent feature gradients, or effectively constant probabilities across a balanced diagnostic set. A single quiet batch does not prove collapse; record a sustained condition over multiple updates. State all numerical guard thresholds in the run configuration.

### D. Checkpoint selection, resume and progress

- Save weights, optimizer, RNG, sampler order/cursor, configuration and source identity in atomic recovery checkpoints. Only claim mid-epoch resume if the sampler cursor and order are restored correctly; otherwise explicitly support epoch-boundary resume.
- Publish progress at a fixed interval: branch, epoch, processed/total examples, remaining examples, throughput, elapsed time, ETA, last checkpoint time and health status. Separate train and validation phases.
- Select checkpoints using a declared validation policy. Ordinary frame accuracy alone is inadequate to establish useful fake-frame detection in this population.
- **Implementation trap:** the existing common trainer treats only `selection == 'accuracy'` as maximization. Merely changing the string to `balanced_accuracy` would not implement the correct objective/direction. Add an explicit metric and optimization direction, with tests, if using balanced accuracy.
- Bind recovery sampler, selection objective and optimization settings into checkpoint identity. Do not let a recovery checkpoint masquerade as the old baseline checkpoint.

### E. Cache compatibility

The current prepared-cache identity includes the whole profile, including training settings. A lower learning rate or changed sampler can therefore make the old identity incompatible even when the raw preprocessing is unchanged.

For the tiny pilot, use a fresh small cache under the recovery namespace. Do not edit old cache identities or bypass checksum checks. Reusing larger old caches requires a separately verified preparation-only identity and input parity; it must not modify old payloads. Check actual free SSD space before expanding preparation to the full population.

## 5. Run a small learning sanity check first

Proposed starting budget, to be recorded before running:

| Branch | Tiny training sample | Initial update budget |
|---|---|---:|
| Audio | 32 genuine-audio + 32 fake-audio clips, prepared successfully | 200 updates |
| Visual | 64 genuine + 64 annotation-modified frames from at least 16 training clips where feasible | 200 updates |

Run the branches separately. These examples may be revisited for the overfit check, but validation clips must not enter training. A balanced sample must be selected after preparation and target validation, not assumed from filenames.

Engineering targets, **not paper results or mandatory statistical thresholds**:

- Finite loss that declines meaningfully on the fixed tiny training set; an initial target is at least a 20% reduction from the initial mean loss.
- Nonconstant scores, nonzero useful feature gradients and an active classifier path.
- Training balanced accuracy clearly above chance; approximately 80% is a useful tiny-set target before spending time on a larger run.

If the tiny-set check fails, stop and inspect activations/gradients. Do not automatically start a multi-hour run or resume the old collapsed checkpoints. A successful tiny-set overfit check demonstrates that learning is possible, not generalization.

## 6. Validate branches independently before full fusion

After the training-only sanity check passes:

1. Start a fresh, separately named pilot with the frozen 3,000 train / 500 validation population and recorded recovery settings. An initial three-epoch budget matches the previous limited run; extend it only from validation evidence and record any changed budget.
2. Evaluate the visual and audio branches independently. Preserve all 500 validation IDs, with explicit unassessed status where necessary.
3. Report audio predictions against audio labels, visual frame predictions against known temporal frame labels, and visual video votes against visual video labels. These are different measurements.
4. Report real/fake score distributions, branch confusion, class recall and branch AUC where continuous probabilities and both classes are available. A fused AUC cannot be invented from hard OR decisions.
5. Run the unchanged majority-vote + hard-OR fusion as the paper-style reference only after both branches show useful validation discrimination.

### Separate issue: short edits versus majority voting

All 500 visually manipulated held-out clips in the existing experiment contain manipulated time below 50%, with a median around 4.5%. A good local frame detector can still produce a majority-real video vote. Fixing branch training does not automatically fix this aggregation limitation.

If a Top-K/peak/localized aggregation is later evaluated, name it as an **AV-Deepfake1M adaptation**, choose all settings on validation and keep the original majority-vote result alongside it. Do not silently alter the reference method or claim exact reproduction.

## 7. Keep the existing held-out set out of recovery selection

Do not run the old 1,000-clip held-out set after every change. Its results have already been inspected, so this recovery is an exploratory follow-up.

Use training and validation only to choose learning rate, epoch, sampler, checkpoint, aggregation and threshold. Lock the complete protocol before a fresh confirmation evaluation. Do not treat the failed AVFakeNet run as proof that the published method is generally inferior to FK.

The earlier `keep-official` population policy reports train/validation source-family overlap. Preserve that documented policy and limitation; do not silently rebuild or relabel the shared dataset during recovery.

## 8. Return these artifacts to Machine A

Create a new folder such as `return-b-recovery-001`. Do not overwrite the existing diagnostic return. Include:

- Recovery configuration, code file hashes, Git revision and any uncommitted patch.
- Frozen tiny-training sample definition and the 3,500-row population hash.
- Environment/package versions, device, seed, preparation settings and actual cache identity.
- Per-branch selected `best.pt`, resumable checkpoint if applicable, `run.json`, all epoch metrics and health/progress logs.
- The tiny-set pass/fail report, including measured criteria rather than only a success message.
- Validation predictions with genuine raw branch probabilities, labels, assessment status and failure reasons; separate frame/window/clip metrics explicitly.
- Selected-checkpoint SHA-256 values and checksummed file inventory.
- A concise `HANDOFF.md`: what changed, what passed, what remains untested, and whether any held-out inference was performed.

The existing `transfer_worker_results.py export` expects the old workflow layout. Do not assume it can package a new custom recovery layout. Either keep a compatible verified structure or implement an exporter for the recovery run and verify every copied file.

## 9. Definition of done for this handoff

Machine B should first deliver **a passing training sanity check and independently verified validation branch diagnostics**. A completed process or high majority-frame accuracy is insufficient.

The larger recovery run may proceed only after the branch-learning health checks pass. Completed historical results must still match their original checksums when recovery work ends.
