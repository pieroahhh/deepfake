
# Portable workspace paths; no filesystem aliases.
$fkRoot = $PSScriptRoot
while ($fkRoot -and -not (Test-Path -LiteralPath (Join-Path $fkRoot 'workspace-layout.json'))) { $fkRoot = Split-Path -Parent $fkRoot }
if (-not $fkRoot) { throw 'workspace-layout.json is missing' }
. (Join-Path $fkRoot '06_Tools\workspace\workspace_paths.ps1')
$script:FK_WorkspaceProject = 'main'
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$log=Join-FKPath $repo (Resolve-FKPath 'runs\remaining-reboot-recovery.log')
try {
    $target=Join-FKPath $repo (Resolve-FKPath 'runs\remaining-ablations-001\no_av')
    if(Test-Path -LiteralPath $target){
        if(-not(Test-Path (Join-FKPath $target 'completed.json'))){
            if(Test-Path (Join-FKPath $target 'attempt-001\last.pt')){throw 'Saved checkpoint exists; explicit resume required'}
            $destination=Join-FKPath (Split-Path $target -Parent) ('no_av-interrupted-'+(Get-Date -Format 'yyyyMMdd-HHmmss'))
            [System.IO.Directory]::Move($target,$destination)
            "Preserved interrupted first epoch: $destination" | Add-Content $log
        }
    }
    foreach($name in @('FK-Remaining-Core-Ablations-001','FK-Remaining-Ablation-Evaluation-001','FK-Remaining-Derivatives-001','FK-Remaining-Supplement-001')){
        $task=Get-ScheduledTask -TaskName $name
        if($task.State -ne 'Running'){Start-ScheduledTask -TaskName $name}
        "Started/retained $name at $(Get-Date -Format o)" | Add-Content $log
    }
} catch { $_ | Out-String | Add-Content $log; throw }
