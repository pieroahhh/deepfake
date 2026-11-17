
# Portable workspace paths; no filesystem aliases.
$fkRoot = $PSScriptRoot
while ($fkRoot -and -not (Test-Path -LiteralPath (Join-Path $fkRoot 'workspace-layout.json'))) { $fkRoot = Split-Path -Parent $fkRoot }
if (-not $fkRoot) { throw 'workspace-layout.json is missing' }
. (Join-Path $fkRoot '06_Tools\workspace\workspace_paths.ps1')
$script:FK_WorkspaceProject = 'main'
# Shared setup only; loading this file never starts training.
function Invoke-Checked {
    param([string]$Executable, [Parameter(ValueFromRemainingArguments=$true)][object[]]$CommandArguments)
    & $Executable @CommandArguments
    if($LASTEXITCODE -ne 0) { throw "Command failed: $Executable ($LASTEXITCODE)" }
}
function Install-WorkerEnvironment {
    if(!(Test-Path -LiteralPath (Resolve-FKPath '.venv-gpu/Scripts/python.exe'))) {
        Invoke-Checked uv --cache-dir .cache/uv-gpu venv --python 3.11 .venv-gpu
    }
    Invoke-Checked uv --cache-dir .cache/uv-gpu pip install --python .venv-gpu/Scripts/python.exe -r baselines/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match
    Invoke-Checked uv --cache-dir .cache/uv-gpu pip install --python .venv-gpu/Scripts/python.exe --no-deps -e .
}
