# FK study run

Historical LAV-DF pilot record. The current primary benchmark is AV-Deepfake1M;
follow [the current protocol](avdeepfake1m-paper-alignment.md) for new runs.

The RTX 4060 laptop builds the entire shared cache. Every machine trains from a copy of the finished cache and its manifest. Keep the relative `caches/` layout intact when copying the run directory by USB or LAN.

## Frozen subset

The frozen LAV-DF population has 19,992 clips from 5,406 source families. The split counts are 14,997 train, 2,497 validation, and 2,498 test. The selection keeps the natural modality mixture in the sampled families. Within each official split, rank families by SHA-256 of `"7\0" + source_family`, take the prefix of whole families below the clip cap, and retain every eligible clip in those families. The caps are 15,000, 2,500, and 2,500. The audit records the four out-of-duration annotations and the sampled category counts. [Frozen manifest](../data/lavdf-study-001/manifest.json) · [Audit](../data/lavdf-study-001/audit.json)

The manifest SHA-256 is `760a4bca9debc20b2be0b4aa1f775853977db40eb316ce8337da396d9156dcb0`. The authors describe the official splits as identity-disjoint in Section 3.3. The local audit independently checked source-family separation and hashed the selected media. [LAV-DF paper](https://arxiv.org/html/2305.01979#S3.SS3)

| Split | Genuine | Audio edit only | Visual edit only | Audio and visual edits |
| --- | ---: | ---: | ---: | ---: |
| Train | 4,087 | 3,619 | 3,669 | 3,622 |
| Validation | 660 | 609 | 618 | 610 |
| Test | 659 | 611 | 618 | 610 |

Describe the test set as a seeded 2,498-clip sample of the official LAV-DF test split. Report ROC AUC, balanced accuracy, accuracy, and usable-window coverage by overlap label. The pilot rejected 6 of 98 full windows overlapping an edit and 2 of 233 full windows with zero edit overlap. Compute the study counts from its cache assessments. The proposed training cap is 20 epochs with patience 5 for all four models. Confirm the cap and baseline owners before training. If caption metadata is still inaccessible on 16 November, move the caption comparison to future work.

## Overnight cache

Run from the repo root using the CUDA environment. A fresh run requires a new output path. After interruption, append `--resume` and keep the same output path, manifest, code, assets, device, and thread settings. The cache journal verifies completed payloads and reuses them. It moves an invalid clip cache aside and rebuilds that clip. `training-manifest.json` is published after all clip assessments are complete.

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:OMP_NUM_THREADS='8'
$env:MKL_NUM_THREADS='8'
$fkPython = '.\.cache\fk-cuda-venv\Scripts\python.exe'
$launcher = 'import cv2; from deepfake_detection.cli import main; cv2.setNumThreads(8); raise SystemExit(main())'
& $fkPython -c $launcher cache-fk --manifest data/lavdf-study-001/manifest.json --dataset-root LAV-DF --assets checkpoints/fk-assets-001/assets.local.json --device cuda --output runs/lavdf-study-cache-001
```

The old OpenCV environment setting left OpenCV at 24 threads on this laptop. The launcher sets eight explicitly. On the same 50 training clips, Torch 2 and OpenCV 24 averaged 1.66 s/clip, and Torch 8 and OpenCV 8 averaged 1.58 s/clip, both excluding startup. All clip assessments, usable windows, and feature tensors matched. The eight-thread pass ran second, so warm file reads may explain part of the difference. The 1.58 s/clip result projects to 8.8 hours and keeps the full frozen subset on one laptop. The pilot storage measurement projects to about 75.5 GB. Actual time and size depend on clip duration and usable windows. Use the finished manifest hash and payload SHA-256 checks when confirming the copies. [Profile comparison](../../../04_Results/experiments/fk/fk-study-cache-profile-001/comparison.json)

## Transfer the verified FK assets

The transfer bundle is `runs/fk-study-asset-transfer-001/checkpoints/fk-assets-001` on the caching laptop. Copy that folder under `checkpoints/` in each teammate's checkout. The included `assets.portable.json` resolves model paths from the current directory, so run FK commands from the repository root. Keep the four model directories and their manifests intact. The original laptop registry is `assets.local.json`.

Verify all four assets in the machine's CUDA environment before running media inference or caching. Run this from the repository root with that environment's Python.

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
@'
import json
from deepfake_detection.cli import load_json
from deepfake_detection.commands import fk_asset_hashes, load_fk_assets

assets = load_json('checkpoints/fk-assets-001/assets.portable.json')
encoders, _, _ = load_fk_assets(assets)
print(json.dumps(fk_asset_hashes(assets, encoders), indent=2))
'@ | python -
```

## Section III-E teammate handoff

| Workstream | Input from each shared cache |
| --- | --- |
| Video-only | `visual_features`, `visual_times`, and the shared usable window intervals |
| Audio-only | `audio_features`, `audio_times`, and the shared usable window intervals |
| Concatenation fusion | Both frozen feature streams aligned to the same windows or temporal bins before concatenation |

Develop each baseline on the 48-clip manifest first. Use the same clip IDs, usable windows, official splits, and combined authenticity target as FK. Select thresholds on validation and retain run configs, checkpoints, and scored predictions. Report usable-window coverage by clip category and window label. Keep each baseline’s architecture and training budget explicit so the paper comparison can be reproduced. [Pilot cache manifest](../../../04_Results/experiments/fk/lavdf-fk-cache-001/training-manifest.json)

On each GPU machine, verify the copied cache against its hashes and run a CUDA window and checkpoint resume smoke test before full training. Use `--device cuda` for measured runs. Check the Silero Python 3.14 warning on any machine that will run media inference. Read `metrics.json` for completed FK training status. Reserve official test scoring until the protocol and validation choices are frozen.
