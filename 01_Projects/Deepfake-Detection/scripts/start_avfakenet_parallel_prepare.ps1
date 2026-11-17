
# Portable workspace paths; no filesystem aliases.
$fkRoot = $PSScriptRoot
while ($fkRoot -and -not (Test-Path -LiteralPath (Join-Path $fkRoot 'workspace-layout.json'))) { $fkRoot = Split-Path -Parent $fkRoot }
if (-not $fkRoot) { throw 'workspace-layout.json is missing' }
. (Join-Path $fkRoot '06_Tools\workspace\workspace_paths.ps1')
$script:FK_WorkspaceProject = 'main'
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-FKPath $PSScriptRoot '..')).Path
$run = Join-FKPath $repo (Resolve-FKPath 'runs\avfakenet-normalized-pilot-001')
if (Test-Path -LiteralPath (Join-FKPath $run 'pause.request')) { throw 'A pause request exists. Wait for PAUSED and remove that request before resuming.' }
$python = Join-FKPath $repo (Resolve-FKPath '.venv-gpu\Scripts\python.exe')
$script = Join-FKPath $repo 'scripts\parallel_avfakenet_prepare.py'
$evidence = Join-FKPath $repo (Resolve-FKPath 'runs\avfakenet-preparation-benchmark-001')
$dataset = (Resolve-Path (Join-FKPath $repo '..\avh-a\dataset')).Path
$env:HF_HOME = Join-FKPath $repo (Resolve-FKPath '.cache\huggingface')
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$arguments = @('-u', ('"'+$script+'"'), '--run', ('"'+$run+'"'), '--dataset-root', ('"'+$dataset+'"'), '--evidence', ('"'+$evidence+'"'), '--reuse-decoder')
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput (Join-FKPath $run ('parallel-'+$stamp+'.log')) -RedirectStandardError (Join-FKPath $run ('parallel-'+$stamp+'.err.log')) -PassThru
@{pid=$process.Id;started=(Get-Date -Format o);run=$run;mode='parallel preparation then original pilot'} | ConvertTo-Json | Set-Content -LiteralPath (Join-FKPath $run 'parallel-launcher.json') -Encoding UTF8
Write-Host ('Started preparation accelerator process {0}. Check progress.json for actual state.' -f $process.Id)
