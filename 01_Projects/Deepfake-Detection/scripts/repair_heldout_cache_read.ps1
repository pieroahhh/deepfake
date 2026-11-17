# One-time elevated permission repair explicitly authorized by the user.
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Split-Path $PSScriptRoot -Parent))
$target=[IO.Path]::GetFullPath((Join-Path $repo 'runs\heldout-001\fk\cache'))
if($target -ne 'C:\Users\teeku\Desktop\fk\Deepfake-Detection\runs\heldout-001\fk\cache'){throw 'Unexpected repair target'}
$out=Join-Path $repo 'runs\heldout-cache-read-repair-001'
[IO.Directory]::CreateDirectory($out) | Out-Null
try {
    $principal=New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    if(-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)){throw 'Administrator token required for the one-time repair'}
    $manifest=Join-Path $target 'training-manifest.json'
    $before=(Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash
    & icacls.exe $target /save (Join-Path $out 'acl-before.txt') /T /C *> (Join-Path $out 'acl-backup.log')
    if($LASTEXITCODE -ne 0){throw 'ACL backup failed; no permission changes applied'}
    # Add current user read/traverse only; retain existing grants and ownership.
    & icacls.exe $target /grant '*S-1-5-21-1502947017-976547090-1717846262-1001:(OI)(CI)(RX)' /T /C *> (Join-Path $out 'acl-repair.log')
    if($LASTEXITCODE -ne 0){throw 'Some ACL grants failed; inspect acl-repair.log'}
    $after=(Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash
    if($before -ne $after){throw 'Manifest content changed'}
    @{status='FINISHED';target=$target;manifest_sha256=$after;manifest_unchanged=$true;ownership_changed=$false;grant='Current user read and traverse';finished=(Get-Date -Format o)} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out 'result.json') -Encoding UTF8
} catch {
    @{status='FAILED';target=$target;error=$_.Exception.Message;finished=(Get-Date -Format o)} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out 'result.json') -Encoding UTF8
    throw
}
