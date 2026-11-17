param([switch]$Setup, [switch]$Run)
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
$env:HF_HOME = Join-FKPath $RepoPath (Resolve-FKPath '.cache/huggingface-worker-b')
if($Run) {
    Invoke-Checked $PythonPath scripts/verify_second_machine.py --bundle $BundlePath --role B
    Invoke-Checked $PythonPath scripts/check_worker_cuda.py
    Invoke-Checked $PythonPath scripts/run_two_day.py --methods avfakenet --manifest runs/two-day-001/population/manifest.json --dataset-root $DatasetPath --output runs/worker-b/avfakenet --threads 4
    Invoke-Checked $PythonPath scripts/transfer_worker_results.py export --worker-root runs/worker-b --role B --output (Join-FKPath $BundlePath 'return-b-to-machine-a')
}
