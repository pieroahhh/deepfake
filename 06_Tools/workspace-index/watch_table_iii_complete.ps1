
# Portable workspace paths; no filesystem aliases.
$fkRoot = $PSScriptRoot
while ($fkRoot -and -not (Test-Path -LiteralPath (Join-Path $fkRoot 'workspace-layout.json'))) { $fkRoot = Split-Path -Parent $fkRoot }
if (-not $fkRoot) { throw 'workspace-layout.json is missing' }
. (Join-Path $fkRoot '06_Tools\workspace\workspace_paths.ps1')
$script:FK_WorkspaceProject = 'main'
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$resultPath = Join-FKPath $taskRoot 'Deepfake-Detection\runs\table-iii-complete-001'
$progressPath = Join-FKPath $resultPath 'progress.json'
Write-Host 'Checking every 3 seconds. Ctrl+C stops this viewer; the background worker continues.'
while ($true) {
    if (Test-Path -LiteralPath $progressPath) {
        try {
            $state = Get-Content -LiteralPath $progressPath -Raw | ConvertFrom-Json
            Write-Host ("{0} | {1} | {2}/{3} ({4}%) | {5}s" -f $state.status,$state.stage,$state.processed,$state.total,$state.percent,$state.elapsed_seconds)
            if ($state.status -in @('FINISHED','FAILED')) {
                if ($state.error) { Write-Host $state.error -ForegroundColor Red }
                if ($state.publication) { Write-Host "Results: $($state.publication)" }
                break
            }
        } catch { Write-Host 'Waiting for the next complete progress update...' }
    } else { Write-Host 'Waiting for publisher startup...' }
    $launchPath = Join-FKPath $resultPath 'launcher.json'
    if (Test-Path -LiteralPath $launchPath) {
        $launch = Get-Content -LiteralPath $launchPath -Raw | ConvertFrom-Json
        if (-not (Get-Process -Id $launch.pid -ErrorAction SilentlyContinue)) {
            Write-Host 'The background process has exited. Last error lines:' -ForegroundColor Yellow
            Get-Content -LiteralPath (Join-FKPath $resultPath 'publish.err.log') -Tail 20 -ErrorAction SilentlyContinue
            break
        }
    }
    Start-Sleep -Seconds 3
}
