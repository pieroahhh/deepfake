param(
    [string]$RunRoot = (Join-FKPath $PSScriptRoot '..\runs\avfakenet-normalized-full-tiny-002'),
    [int]$IntervalSeconds = 3
)
# Portable workspace paths; no filesystem aliases.
$fkRoot = $PSScriptRoot
while ($fkRoot -and -not (Test-Path -LiteralPath (Join-Path $fkRoot 'workspace-layout.json'))) { $fkRoot = Split-Path -Parent $fkRoot }
if (-not $fkRoot) { throw 'workspace-layout.json is missing' }
. (Join-Path $fkRoot '06_Tools\workspace\workspace_paths.ps1')
$script:FK_WorkspaceProject = 'main'

$ErrorActionPreference = 'Stop'
$progressPath = Join-FKPath $RunRoot 'progress.json'
$previous = ''
while ($true) {
    if (-not (Test-Path -LiteralPath $progressPath)) {
        Write-Host ('Waiting for {0}' -f $progressPath)
        Start-Sleep -Seconds $IntervalSeconds
        continue
    }
    try { $raw = Get-Content -LiteralPath $progressPath -Raw; $state = $raw | ConvertFrom-Json }
    catch { Start-Sleep -Seconds $IntervalSeconds; continue }
    if ($raw -ne $previous) {
        $line = '{0}  {1}  {2}  {3}/{4} updates' -f (Get-Date -Format 'HH:mm:ss'), $state.status, $state.branch, $state.update, $state.updates_per_branch
        if ($null -ne $state.balanced_accuracy) { $line += '  train BA={0:P2}' -f $state.balanced_accuracy }
        if ($null -ne $state.loss) { $line += '  loss={0:F4}' -f $state.loss }
        if ($null -ne $state.branch_eta_seconds) { $line += '  branch ETA={0:N0}s' -f $state.branch_eta_seconds }
        if ($state.sanity_result) { $line += '  sanity={0}' -f $state.sanity_result }
        Write-Host $line
        if ($state.error) { Write-Host $state.error }
        $previous = $raw
    }
    if ($state.status -in @('FINISHED', 'FAILED', 'PAUSED')) { break }
    Start-Sleep -Seconds $IntervalSeconds
}
