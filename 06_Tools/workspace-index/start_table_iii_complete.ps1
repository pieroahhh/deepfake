
# Portable workspace paths; no filesystem aliases.
$fkRoot = $PSScriptRoot
while ($fkRoot -and -not (Test-Path -LiteralPath (Join-Path $fkRoot 'workspace-layout.json'))) { $fkRoot = Split-Path -Parent $fkRoot }
if (-not $fkRoot) { throw 'workspace-layout.json is missing' }
. (Join-Path $fkRoot '06_Tools\workspace\workspace_paths.ps1')
$script:FK_WorkspaceProject = 'main'
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$pythonPath = Join-FKPath $taskRoot 'Deepfake-Detection\.venv-gpu\Scripts\python.exe'
$runnerPath = Join-FKPath $PSScriptRoot 'finish_table_iii_complete.py'
$resultPath = Join-FKPath $taskRoot 'Deepfake-Detection\runs\table-iii-complete-001'
New-Item -ItemType Directory -Path $resultPath -Force | Out-Null
$launchPath = Join-FKPath $resultPath 'launcher.json'
if (Test-Path -LiteralPath $launchPath) {
    $previous = Get-Content -LiteralPath $launchPath -Raw | ConvertFrom-Json
    if (Get-Process -Id $previous.pid -ErrorAction SilentlyContinue) {
        throw "A previous publisher is still running (PID $($previous.pid)). Use the progress command."
    }
}
$env:PYTHONIOENCODING = 'utf-8'
@{status='STARTING';stage='publisher-startup';processed=0;total=8;percent=0;elapsed_seconds=0} | ConvertTo-Json | Set-Content -LiteralPath (Join-FKPath $resultPath 'progress.json') -Encoding UTF8
$worker = Start-Process -FilePath $pythonPath -ArgumentList @('-u', ('"' + $runnerPath + '"')) -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-FKPath $resultPath 'publish.log') -RedirectStandardError (Join-FKPath $resultPath 'publish.err.log') -PassThru
@{pid=$worker.Id; started=(Get-Date).ToString('o'); runner=$runnerPath} | ConvertTo-Json | Set-Content -LiteralPath $launchPath -Encoding UTF8
Write-Host "Started publisher in the background. PID: $($worker.Id)"
Write-Host "Progress: $resultPath\progress.json"
