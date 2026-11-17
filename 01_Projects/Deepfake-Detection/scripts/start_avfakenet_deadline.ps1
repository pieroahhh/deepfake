
# Portable workspace paths; no filesystem aliases.
$fkRoot = $PSScriptRoot
while ($fkRoot -and -not (Test-Path -LiteralPath (Join-Path $fkRoot 'workspace-layout.json'))) { $fkRoot = Split-Path -Parent $fkRoot }
if (-not $fkRoot) { throw 'workspace-layout.json is missing' }
. (Join-Path $fkRoot '06_Tools\workspace\workspace_paths.ps1')
$script:FK_WorkspaceProject = 'main'
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-FKPath $PSScriptRoot '..')).Path
$run = Join-FKPath $repo (Resolve-FKPath 'runs\avfakenet-normalized-pilot-001')
if (Test-Path -LiteralPath (Join-FKPath $run 'pause.request')) { throw 'Wait for PAUSED and remove acknowledged pause.request before starting.' }
$env:HF_HOME = Join-FKPath $repo (Resolve-FKPath '.cache\huggingface')
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$script = Join-FKPath $repo 'scripts\run_avfakenet_deadline.py'
$python = Join-FKPath $repo (Resolve-FKPath '.venv-gpu\Scripts\python.exe')
$process = Start-Process -FilePath $python -ArgumentList @('-u',('"'+$script+'"')) -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput (Join-FKPath $run ('deadline-'+$stamp+'.log')) -RedirectStandardError (Join-FKPath $run ('deadline-'+$stamp+'.err.log')) -PassThru
@{pid=$process.Id;started=(Get-Date -Format o);mode='six-worker preparation then one-epoch pilot'} | ConvertTo-Json | Set-Content -LiteralPath (Join-FKPath $run 'deadline-launcher.json') -Encoding UTF8
Write-Host ('Started process {0}; same population, one epoch per branch.' -f $process.Id)
