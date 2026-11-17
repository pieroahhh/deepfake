
# Portable workspace paths; no filesystem aliases.
$fkRoot = $PSScriptRoot
while ($fkRoot -and -not (Test-Path -LiteralPath (Join-Path $fkRoot 'workspace-layout.json'))) { $fkRoot = Split-Path -Parent $fkRoot }
if (-not $fkRoot) { throw 'workspace-layout.json is missing' }
. (Join-Path $fkRoot '06_Tools\workspace\workspace_paths.ps1')
$script:FK_WorkspaceProject = 'main'
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-FKPath $PSScriptRoot '..')).Path
$run = Join-FKPath $repo (Resolve-FKPath 'runs\avfakenet-normalized-pilot-001')
$target = Join-FKPath $repo (Resolve-FKPath 'runs\avfakenet-one-epoch-001')
$cache = Join-FKPath $repo (Resolve-FKPath 'runs\avfakenet-machine-b-prepared-001')
$env:HF_HOME = Join-FKPath $repo (Resolve-FKPath '.cache\huggingface')
Set-Location -LiteralPath $repo
$python = Join-FKPath $repo (Resolve-FKPath '.venv-gpu\Scripts\python.exe')
& $python -u (Join-FKPath $repo 'scripts\pack_machine_b_cache.py')
if ($LASTEXITCODE -ne 0) { throw 'Cache packing failed; training not started.' }
@{ output=$target; epochs_per_branch=1; prepared_cache=$cache; reason='Verified complete homogeneous Machine B cache'; automatic_heldout=$false } | ConvertTo-Json | Set-Content -LiteralPath (Join-FKPath $run 'next-run.json') -Encoding UTF8
& $python -u (Join-FKPath $repo 'scripts\run_avfakenet_one_epoch.py') --sanity (Join-FKPath $repo (Resolve-FKPath 'runs\avfakenet-normalized-full-tiny-004')) --bundle 'C:\Users\teeku\Downloads\LANDrop\return-b-recovery-probe-inputs-001' --dataset-root (Join-FKPath $repo '..\avh-a\dataset') --output $target --prepared-cache $cache --resume
if ($LASTEXITCODE -ne 0) { throw 'One-epoch pilot stopped; inspect log and progress.' }
