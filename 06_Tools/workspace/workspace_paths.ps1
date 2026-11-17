function Resolve-FKPath {
    param([Parameter(Mandatory,Position=0)][string]$Value, [string]$Project = $script:FK_WorkspaceProject)
    if (-not $script:FK_WorkspaceRoot) {
        $candidate = $PSScriptRoot
        while ($candidate -and -not (Test-Path -LiteralPath (Microsoft.PowerShell.Management\Join-Path $candidate 'workspace-layout.json'))) { $candidate = Split-Path -Parent $candidate }
        if (-not $candidate) { throw 'workspace-layout.json is missing' }
        $script:FK_WorkspaceRoot = $candidate
        $script:FK_WorkspaceLayout = Get-Content -LiteralPath (Microsoft.PowerShell.Management\Join-Path $candidate 'workspace-layout.json') -Raw | ConvertFrom-Json
    }
    $layout = $script:FK_WorkspaceLayout
    if ($layout.state -ne 'migrated') { return $Value }
    $root = $script:FK_WorkspaceRoot.Replace('\','/').TrimEnd('/')
    $text = $Value.Replace('\','/')
    $absolute = [IO.Path]::IsPathRooted($Value)
    if ($absolute) {
        $text = [IO.Path]::GetFullPath($Value).Replace('\','/')
        if (-not $text.StartsWith($root + '/', [StringComparison]::OrdinalIgnoreCase)) { return $Value }
        $relative = $text.Substring($root.Length + 1)
    } else { $relative = $text -replace '^\./', '' }
    $candidates = @($relative)
    if ($relative.StartsWith('01_Projects/') -or $relative.StartsWith('06_Tools/')) { $candidates += $relative.Substring($relative.IndexOf('/') + 1) }
    if (-not $absolute -and $Project -and $relative -match '^(runs|checkpoints|\.cache|\.venv(?:-gpu|-baselines)?)(/|$)') {
        $candidates += $layout.original_project_roots.$Project + '/' + $relative
    }
    foreach ($candidate in $candidates) {
        foreach ($entry in ($layout.path_mappings.PSObject.Properties | Sort-Object { $_.Name.Length } -Descending)) {
            $old = $entry.Name
            if ($candidate.Equals($old,[StringComparison]::OrdinalIgnoreCase) -or $candidate.StartsWith($old + '/', [StringComparison]::OrdinalIgnoreCase)) {
                return [IO.Path]::GetFullPath($root + '/' + $entry.Value + $candidate.Substring($old.Length))
            }
        }
    }
    return $Value
}

function Join-FKPath {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory,Position=0)][string]$Path,
        [Parameter(Mandatory,Position=1)][string]$ChildPath,
        [Parameter(ValueFromRemainingArguments=$true)][string[]]$AdditionalChildPath,
        [switch]$Resolve
    )
    if ([IO.Path]::IsPathRooted($ChildPath)) { $joined = $ChildPath }
    else { $joined = Microsoft.PowerShell.Management\Join-Path -Path $Path -ChildPath $ChildPath }
    foreach ($child in $AdditionalChildPath) { $joined = Microsoft.PowerShell.Management\Join-Path -Path $joined -ChildPath $child }
    $result = Resolve-FKPath $joined
    if ($Resolve) { return (Resolve-Path -LiteralPath $result).Path }
    return $result
}
