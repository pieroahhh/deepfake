param(
    [Parameter(Mandatory=$true)][string]$DatasetRoot,
    [Parameter(Mandatory=$true)][string]$RunRoot,
    [Parameter(Mandatory=$true)][string[]]$CommandArguments,
    [string]$Image = 'fk-avhubert:cpu-v1'
)
$ErrorActionPreference = 'Stop'
$Repository = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$Dataset = (Resolve-Path -LiteralPath $DatasetRoot).Path
$Runs = (Resolve-Path -LiteralPath $RunRoot).Path
# Explicit mount paths make commands portable between Windows and Linux.
& docker run --rm --mount "type=bind,source=$Repository,target=/workspace,readonly" --mount "type=bind,source=$Dataset,target=/dataset,readonly" --mount "type=bind,source=$Runs,target=/runs" $Image @CommandArguments
if ($LASTEXITCODE -ne 0) { throw "AVTENet Docker worker exited with $LASTEXITCODE" }
