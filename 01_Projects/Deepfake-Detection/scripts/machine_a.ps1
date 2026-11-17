param(
    [ValidateSet('FK','Compare')][string]$Stage = 'FK',
    [string]$DatasetRoot = 'D:\AV-Deepfake1M',
    [string]$ReturnedB,
    [string]$ReturnedC
)
# Portable workspace paths; no filesystem aliases.
$fkRoot = $PSScriptRoot
while ($fkRoot -and -not (Test-Path -LiteralPath (Join-Path $fkRoot 'workspace-layout.json'))) { $fkRoot = Split-Path -Parent $fkRoot }
if (-not $fkRoot) { throw 'workspace-layout.json is missing' }
. (Join-Path $fkRoot '06_Tools\workspace\workspace_paths.ps1')
$script:FK_WorkspaceProject = 'main'

$ErrorActionPreference = 'Stop'
$RepoPath = (Resolve-Path (Join-FKPath $PSScriptRoot '..')).Path
Set-Location -LiteralPath $RepoPath
$PythonPath = Join-FKPath $RepoPath (Resolve-FKPath '.venv-gpu/Scripts/python.exe')
function Invoke-Checked {
    param([string]$Executable, [Parameter(ValueFromRemainingArguments=$true)][object[]]$CommandArguments)
    & $Executable @CommandArguments
    if($LASTEXITCODE -ne 0) { throw "Command failed: $Executable ($LASTEXITCODE)" }
}
if($Stage -eq 'FK') {
    Invoke-Checked $PythonPath scripts/run_machine_a.py --dataset-root $DatasetRoot
}
if($Stage -eq 'Compare') {
    if($ReturnedB) {
        Invoke-Checked $PythonPath scripts/transfer_worker_results.py import --bundle $ReturnedB --output runs/two-day-001/incoming-b --expected-role B --manifest runs/two-day-001/population/manifest.json
    }
    if($ReturnedC) {
        Invoke-Checked $PythonPath scripts/transfer_worker_results.py import --bundle $ReturnedC --output runs/two-day-001/incoming-c --expected-role C --manifest runs/two-day-001/population/manifest.json
    }
    Invoke-Checked $PythonPath scripts/compare_three_models.py --manifest runs/two-day-001/population/manifest.json --fk runs/two-day-001/experiment/fk/validation.json --avfakenet runs/two-day-001/incoming-b/avfakenet/evaluation/predictions.jsonl --avtenet runs/two-day-001/incoming-c/avtenet/evaluation/predictions.jsonl --output runs/two-day-001/three-model-comparison.json
}
