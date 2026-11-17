
# Portable workspace paths; no filesystem aliases.
$fkRoot = $PSScriptRoot
while ($fkRoot -and -not (Test-Path -LiteralPath (Join-Path $fkRoot 'workspace-layout.json'))) { $fkRoot = Split-Path -Parent $fkRoot }
if (-not $fkRoot) { throw 'workspace-layout.json is missing' }
. (Join-Path $fkRoot '06_Tools\workspace\workspace_paths.ps1')
$script:FK_WorkspaceProject = 'main'
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-FKPath $PSScriptRoot '..')).Path
$run = Join-FKPath $repo (Resolve-FKPath 'runs\avfakenet-normalized-pilot-001')
if (Test-Path -LiteralPath (Join-FKPath $run 'pause.request')) { throw 'Wait for PAUSED and remove the acknowledged pause.request before resuming.' }
$evidence = Join-FKPath $repo (Resolve-FKPath 'runs\avfakenet-worker-scaling-001')
$report = Get-Content -LiteralPath (Join-FKPath $evidence 'report.json') -Raw | ConvertFrom-Json
if ($report.status -ne 'COMPLETE') { throw 'Worker-scaling benchmark is incomplete.' }
$python = Join-FKPath $repo (Resolve-FKPath '.venv-gpu\Scripts\python.exe')
$script = Join-FKPath $repo 'scripts\run_avfakenet_resilient.py'
$dataset = (Resolve-Path (Join-FKPath $repo '..\avh-a\dataset')).Path
$env:HF_HOME = Join-FKPath $repo (Resolve-FKPath '.cache\huggingface')
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$arguments = @('-u', ('"'+$script+'"'), '--mode', 'prepare', '--run', ('"'+$run+'"'), '--dataset-root', ('"'+$dataset+'"'), '--evidence', ('"'+$evidence+'"'))
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput (Join-FKPath $run ('scaled-'+$stamp+'.log')) -RedirectStandardError (Join-FKPath $run ('scaled-'+$stamp+'.err.log')) -PassThru
@{pid=$process.Id;started=(Get-Date -Format o);workers=$report.selected_workers;mode='measured worker scaling then unchanged pilot'} | ConvertTo-Json | Set-Content -LiteralPath (Join-FKPath $run 'scaled-launcher.json') -Encoding UTF8
Write-Host ('Started process {0}, selected {1} workers. Check progress.json.' -f $process.Id,$report.selected_workers)
