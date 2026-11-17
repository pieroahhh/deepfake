param([switch]$Setup, [switch]$BuildContainer, [switch]$Run)
# Portable workspace paths; no filesystem aliases.
$fkRoot = $PSScriptRoot
while ($fkRoot -and -not (Test-Path -LiteralPath (Join-Path $fkRoot 'workspace-layout.json'))) { $fkRoot = Split-Path -Parent $fkRoot }
if (-not $fkRoot) { throw 'workspace-layout.json is missing' }
. (Join-Path $fkRoot '06_Tools\workspace\workspace_paths.ps1')
$script:FK_WorkspaceProject = 'avhubert'

$ErrorActionPreference = 'Stop'
$RepoPath = (Resolve-Path (Join-FKPath $PSScriptRoot '..')).Path
$BundlePath = (Resolve-Path (Join-FKPath $RepoPath '..')).Path
$DatasetPath = Join-FKPath $BundlePath 'dataset'
Set-Location -LiteralPath $RepoPath
. (Join-FKPath $PSScriptRoot 'worker_environment.ps1')
if($Setup) { Install-WorkerEnvironment }
$PythonPath = Join-FKPath $RepoPath (Resolve-FKPath '.venv-gpu/Scripts/python.exe')
$env:HF_HOME = Join-FKPath $RepoPath (Resolve-FKPath '.cache/huggingface-worker-c')
if($BuildContainer) {
    Invoke-Checked docker build --build-arg AVHUBERT_COMMIT=258fb50e155134eec2c4b49c2ae8de267075fd18 --build-arg TORCH_FLAVOR=cu117 -t fk-avhubert:cuda117-v1 baselines/avtenet/runtime
}
if($Run) {
    Invoke-Checked $PythonPath scripts/verify_second_machine.py --bundle $BundlePath --role C
    Invoke-Checked $PythonPath scripts/check_worker_cuda.py
    Invoke-Checked docker run --rm --gpus all --mount "type=bind,source=$RepoPath,target=/workspace,readonly" --entrypoint python fk-avhubert:cuda117-v1 /workspace/scripts/check_worker_cuda.py
    Invoke-Checked $PythonPath scripts/configure_worker_c.py
    Invoke-Checked $PythonPath scripts/run_avtenet_subset.py --stages 'audio,visual,audiovisual,fusion' --index runs/two-day-001/avtenet/shared-index --manifest runs/two-day-001/population/manifest.json --dataset-root $DatasetPath --assets runs/worker-c/assets.local.json --docker-assets runs/worker-c/assets.docker.local.json --output runs/worker-c/avtenet --epochs 1 --threads 4
    Invoke-Checked $PythonPath scripts/transfer_worker_results.py export --worker-root runs/worker-c --role C --output (Join-FKPath $BundlePath 'return-c-to-machine-a')
}
