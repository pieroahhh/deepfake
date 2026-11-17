param([int]$IntervalSeconds = 5, [string]$RunRoot = '')
# Portable workspace paths; no filesystem aliases.
$fkRoot = $PSScriptRoot
while ($fkRoot -and -not (Test-Path -LiteralPath (Join-Path $fkRoot 'workspace-layout.json'))) { $fkRoot = Split-Path -Parent $fkRoot }
if (-not $fkRoot) { throw 'workspace-layout.json is missing' }
. (Join-Path $fkRoot '06_Tools\workspace\workspace_paths.ps1')
$script:FK_WorkspaceProject = 'main'

$ErrorActionPreference = 'Stop'
$run = Join-FKPath $PSScriptRoot '..\runs\avfakenet-normalized-pilot-001'
if ($RunRoot) { $run = $RunRoot }
$previous = ''
$heartbeat = Get-Date
function Read-SharedText([string]$Path) {
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, ([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
    try {
        $reader = New-Object System.IO.StreamReader($stream)
        try { return $reader.ReadToEnd() } finally { $reader.Dispose() }
    } finally { $stream.Dispose() }
}
while ($true) {
    $nextPath = Join-FKPath $run 'next-run.json'
    if (Test-Path -LiteralPath $nextPath) {
        try {
            $next = (Read-SharedText $nextPath) | ConvertFrom-Json
            if (Test-Path -LiteralPath (Join-FKPath $next.output 'progress.json')) { $run = $next.output; $previous = '' }
        } catch { }
    }
    try { $raw=Read-SharedText (Join-FKPath $run 'progress.json'); $state=$raw | ConvertFrom-Json }
    catch { Start-Sleep -Seconds $IntervalSeconds; continue }
    $now=Get-Date
    if ($raw -ne $previous -or ($now-$heartbeat).TotalSeconds -ge 60) {
        $message='{0}  {1}  {2}: {3}/{4} ({5:N2}%)' -f $now.ToString('HH:mm:ss'),$state.status,$state.stage,$state.processed,$state.total,$state.percent
        if ($null -ne $state.eta_seconds -and $state.elapsed_seconds_this_session -ge 120 -and $state.processed -ge 20) {
            $eta=[TimeSpan]::FromSeconds([math]::Max(0,$state.eta_seconds))
            $finish=[DateTimeOffset]::FromUnixTimeSeconds([long]$state.estimated_completion_unix).LocalDateTime
            $message+='  stage ETA {0} / finish ~{1}' -f $eta.ToString('d\.hh\:mm\:ss'),$finish.ToString('MM-dd HH:mm')
        }
        elseif ($state.status -eq 'RUNNING') { $message+='  ETA warming up (cache reuse can distort the initial rate)' }
        if ($state.updated_unix) {
            $age=[DateTimeOffset]::Now.ToUnixTimeSeconds()-[long]$state.updated_unix
            $message+='  last progress {0}s ago' -f $age
            if ($age -ge 180 -and $state.status -eq 'RUNNING') { $message+='  WARNING: stale progress; verify worker/logs (RUNNING is the last saved state)' }
        }
        Write-Host $message
        if ($state.error) { Write-Host $state.error }
        $previous=$raw;$heartbeat=$now
    }
    if ($state.status -in @('FINISHED','FAILED','PAUSED')) { break }
    Start-Sleep -Seconds $IntervalSeconds
}
