# AVTENet validation-only threshold analysis

The diagnostic entry point is `scripts/analyze_avtenet_results.py`. It reads completed model artifacts, verifies their checksums and population labels, reproduces validation clip scores from the frozen Fusion cache/checkpoint on CPU, fits a validation-only balanced-accuracy threshold, freezes the selection, and then applies it to held-out scores. No model is retrained.

Example from the repository root:

```powershell
& .\.venv-gpu\Scripts\python.exe scripts/analyze_avtenet_results.py --code-root "C:\path\to\completed-code" --output "C:\path\to\new-analysis"
```

The output must be outside the completed checkout. It includes `threshold-selection.json`, `report.json`, `heldout-calibrated-predictions.jsonl`, `ANALYSIS.md`, PNG/PDF diagnostic plots and `source-preservation.json`. Use `--resume-analysis` to finish publication after an interrupted analysis; existing frozen selection, reports and predictions must match, and are not replaced. Dependencies are the model environment plus the `reports` optional dependencies in pyproject.toml.

## Machine A findings, 16 November 2026

- All 500 validation clip scores were reproduced; maximum CPU-versus-saved score difference was 2.28e-7. Native real-logit to external fake-score direction passed numerical checks.
- Recomputed validation window accuracy was 91.5045%, matching the completed training log. Clip-level mean aggregation is a different measurement.
- Validation false-negative fake clips had a median known manipulated-window fraction of 27.5%; true positives had 56.3%. This supports mean-score dilution as a contributor, without proving it is the only cause.
- Validation-selected threshold: 0.03562075711254563, maximizing balanced accuracy.
- Held-out accuracy: 35.32% original versus 71.40% calibrated; recall 13.35% versus 64.15%; F1 23.56% versus 77.00%.
- AUC remains 0.840845 and coverage remains 923/1,000. Scores, weights and aggregation did not change.

These are exploratory follow-up results because the original held-out metrics had already been inspected. Threshold fitting itself accepts validation rows only. Keep the original result visible and label the calibrated decision separately. Any revised method needs fresh confirmation. A fair calibrated comparison needs the same validation selection policy for every model and comparison on common assessed clips, alongside all-clip coverage. AVFakeNet diagnostics remain the next step before the final three-model comparison.

The local completed analysis is `runs/avtenet-analysis-001/ANALYSIS.md`. Four unit tests cover validation-only selection, ties, invalid inputs, denominator preservation and input immutability.
