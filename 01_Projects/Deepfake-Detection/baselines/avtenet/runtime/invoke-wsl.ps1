param(
    [Parameter(Mandatory=$true)][string]$Distribution,
    [Parameter(Mandatory=$true)][string]$LinuxPython,
    [Parameter(Mandatory=$true)][string]$LinuxRepository,
    [Parameter(Mandatory=$true)][string[]]$CommandArguments
)
$ErrorActionPreference = 'Stop'
# CommandArguments must use Linux paths (/mnt/d/...); no shell-built command text.
& wsl.exe --distribution $Distribution --cd $LinuxRepository --exec env "PYTHONPATH=$LinuxRepository/src`:$LinuxRepository" $LinuxPython -m baselines.avtenet @CommandArguments
if ($LASTEXITCODE -ne 0) { throw "AVTENet Linux worker exited with $LASTEXITCODE" }
