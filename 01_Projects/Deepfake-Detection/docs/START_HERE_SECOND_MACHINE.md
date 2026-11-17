# Start here: Machine B — AVFakeNet only

This is the **three-machine** workflow. Machine A trains FK, this laptop trains
AVFakeNet, and Machine C runs the complete AVTENet adaptation. This guide replaces
the old two-machine assignment. Machine B no longer runs AV-HuBERT or needs Docker.

## 1. Copy the correct bundle

Copy the entire **Machine-B-bundle** folder to the laptop's SSD. It contains the
current code snapshot, the exact 3,000 train / 500 validation clips, manifest,
audit and shared index. Keep `dataset` beside `code`:

```text
Machine-B-bundle/
  START_HERE_SECOND_MACHINE.md
  bundle.json
  dataset/train/...
  dataset/val/...
  code/scripts/machine_b.ps1
  code/runs/two-day-001/population/...
```

The bundle is about 1.2 GiB. Allow **at least 80 GiB free space** for Python,
prepared face/audio caches, checkpoints and results. Total SSD capacity does not
prove that this much space is free. Do not copy another machine's virtual environment.
Git cloning alone does not supply the dataset, frozen population or shared index.

## 2. Install prerequisites

Use the i7-13700HX / 16 GB RAM / RTX 4060 8 GB laptop with Windows, an NVIDIA
driver, Python **3.11**, and **uv** available in PowerShell. Internet is needed for
dependencies and the pretrained MTCNN face detector on its first use. Plug in AC
power, close large applications, keep the laptop ventilated and prevent sleep.

## 3. Set up and verify

Open PowerShell. Replace the example location with the bundle's actual path:

```powershell
Set-Location -LiteralPath 'D:\Machine-B-bundle\code'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\machine_b.ps1 -Setup
& '.\.venv-gpu\Scripts\python.exe' scripts/verify_second_machine.py --bundle '..' --role B
& '.\.venv-gpu\Scripts\python.exe' scripts/check_worker_cuda.py
```

Wait for setup to finish. Verification must end with `Bundle verified: 3500 clips`.
The CUDA check must identify the RTX 4060 and report `arithmetic_check: PASSED`.
Fix missing/corrupted files by copying them from the original bundle; do not edit
hashes to bypass verification. No system-wide execution-policy change is needed.

## 4. Run one copy of the launcher

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\machine_b.ps1 -Run
```

It verifies the B bundle and CUDA, then runs **visual branch 3 epochs → audio
branch 3 epochs → validation evaluation → result export**. Batch size is 8 and
seed is 7. All 500 validation IDs remain in evaluation, with unavailable cases
reported through coverage. The two-day budget does not guarantee convergence.

Keep this window open. Do not change code, profiles, dependencies or media while
running or resuming. Run only one launcher on this GPU.

## 5. Monitor progress

In a second PowerShell window, inside the same `code` directory:

```powershell
Get-Content '.\runs\worker-b\avfakenet\status.json'
Get-Content '.\runs\worker-b\avfakenet\logs\avfakenet-visual.log' -Tail 20 -Wait
```

Follow the `log` path named by the status file as stages change. Model loading,
video decoding and face detection may take time between updates. Use `nvidia-smi`
to inspect GPU activity. Ctrl+C in the log-viewing window only stops that viewer.

The evaluation files are nested under
`runs/worker-b/avfakenet/avfakenet/evaluation`. The exporter normalizes them to
`avfakenet/evaluation` in the returned folder; do not move intermediate outputs.

## 6. Recover from interruption

Keep `runs/worker-b` and rerun the same `-Run` command after fixing the failed
stage. Completed stages and prepared-input caches are reused. Training resumes
from the last completed epoch into a new attempt directory; an unfinished epoch
is repeated. An attempt without an epoch checkpoint starts from its initial
weights. Preserve `best.pt` beside `last.pt`; they have different purposes.

Partial evaluation directories need to be preserved under another name before
retrying evaluation. Rename them only after the launcher has stopped. For CUDA
memory errors, close other GPU applications and retain the log; coordinate any
batch/profile change before starting a separate experiment. Do not silently
change settings during resume.

If exporting fails because the return folder already exists, check whether the
previous export succeeded. Preserve it under another name before rerunning the
launcher to request a fresh export.

## 7. Return the result

When the launcher succeeds, send the entire folder:

```text
Machine-B-bundle/return-b-to-machine-a/
```

It includes predictions, metrics, selected visual/audio checkpoints and
`return.json` with the B role, population hash and file checksums. The selected
checkpoints retain their run/metric provenance. Dataset, environments, optimizer
resume checkpoints and prepared-input caches are not returned.

Keep your local experiment until Machine A confirms import and comparison.
Machine A imports this B result together with C's completed AVTENet result; see
`code/docs/three-machine-guide.md` for the coordinator commands.

## If you already started the old two-machine bundle

Keep that folder unchanged while it is running. AVFakeNet uses the same dataset,
batch size, seed and epoch budget. Its completed outputs can be retained. Do not
start the old AV-HuBERT stage on B under this new assignment. Coordinate a stop
between stages and update the launcher/return exporter before moving to the new
bundle. Copying a virtual environment to a different location is not supported.
