
# Portable workspace paths; no filesystem aliases.
$fkRoot = $PSScriptRoot
while ($fkRoot -and -not (Test-Path -LiteralPath (Join-Path $fkRoot 'workspace-layout.json'))) { $fkRoot = Split-Path -Parent $fkRoot }
if (-not $fkRoot) { throw 'workspace-layout.json is missing' }
. (Join-Path $fkRoot '06_Tools\workspace\workspace_paths.ps1')
$script:FK_WorkspaceProject = 'main'
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-FKPath $PSScriptRoot '..')).Path
$run = Join-FKPath $repo (Resolve-FKPath 'runs\avfakenet-normalized-pilot-001')
if (-not (Test-Path -LiteralPath (Join-FKPath $run 'identity.json'))) { throw 'Run the gated pilot preflight before using this continuation launcher.' }
if (Test-Path -LiteralPath (Join-FKPath $run 'pause.request')) { throw 'pause.request exists; remove it only when intending to resume.' }
$python = Join-FKPath $repo (Resolve-FKPath '.venv-gpu\Scripts\python.exe')
$script = Join-FKPath $repo 'scripts\run_avfakenet_resilient.py'
$sanity = Join-FKPath $repo (Resolve-FKPath 'runs\avfakenet-normalized-full-tiny-004')
$bundle = 'C:\Users\teeku\Downloads\LANDrop\return-b-recovery-probe-inputs-001'
$dataset = (Resolve-Path (Join-FKPath $repo '..\avh-a\dataset')).Path
$env:HF_HOME = Join-FKPath $repo (Resolve-FKPath '.cache\huggingface')
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$arguments = @('-u', ('"'+$script+'"'), '--mode', 'pilot', '--sanity', ('"'+$sanity+'"'), '--bundle', ('"'+$bundle+'"'), '--dataset-root', ('"'+$dataset+'"'), '--output', ('"'+$run+'"'), '--resume')
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput (Join-FKPath $run ('worker-'+$stamp+'.log')) -RedirectStandardError (Join-FKPath $run ('worker-'+$stamp+'.err.log')) -PassThru
@{ pid=$process.Id; started=(Get-Date -Format o); run=$run } | ConvertTo-Json | Set-Content -LiteralPath (Join-FKPath $run 'launcher.json') -Encoding UTF8
Write-Host ('Started AVFakeNet continuation process {0}. See progress.json for the actual worker state.' -f $process.Id)
