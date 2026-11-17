param([int]$IntervalSeconds=5)
# Portable workspace paths; no filesystem aliases.
$fkRoot = $PSScriptRoot
while ($fkRoot -and -not (Test-Path -LiteralPath (Join-Path $fkRoot 'workspace-layout.json'))) { $fkRoot = Split-Path -Parent $fkRoot }
if (-not $fkRoot) { throw 'workspace-layout.json is missing' }
. (Join-Path $fkRoot '06_Tools\workspace\workspace_paths.ps1')
$script:FK_WorkspaceProject = 'main'

$ErrorActionPreference='Stop'
$run=Join-FKPath $PSScriptRoot '..\runs\remaining-ablations-001'
$last=''
while($true){
    $next=Join-FKPath $PSScriptRoot '..\runs\remaining-ablation-evaluation-001'
    if(Test-Path -LiteralPath (Join-FKPath $run 'queue-complete.json')){
        if(Test-Path -LiteralPath (Join-FKPath $next 'progress.json')){$run=$next;$last=''}
    }
    try{
        $stream=[System.IO.File]::Open((Join-FKPath $run 'progress.json'),[System.IO.FileMode]::Open,[System.IO.FileAccess]::Read,([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
        try{$reader=New-Object System.IO.StreamReader($stream);try{$raw=$reader.ReadToEnd()}finally{$reader.Dispose()}}finally{$stream.Dispose()}
        $state=$raw|ConvertFrom-Json
    }catch{Start-Sleep -Seconds $IntervalSeconds;continue}
    $key='{0}|{1}|{2}|{3}|{4}|{5}' -f $state.status,$state.variant,$state.stage,$state.epoch,$state.processed,$state.completed_variants.Count
    if($key -ne $last){
        $text='{0} {1} | {2} | {3} | completed variants {4}/{5}' -f (Get-Date -Format 'HH:mm:ss'),$state.status,$state.variant,$state.stage,$state.completed_variants.Count,$state.variants.Count
        if($state.epoch){$text+=' | epoch {0}/{1}' -f $state.epoch,$state.epochs}
        if($state.total){$text+=' | clips {0}/{1} ({2:N1}%)' -f $state.processed,$state.total,(100*$state.processed/$state.total)}
        if($state.total -and $state.processed -gt 0 -and $state.last_log_line -match 'elapsed ([0-9.]+) min'){
            $elapsed=[double]::Parse($Matches[1],[System.Globalization.CultureInfo]::InvariantCulture)*60
            if($elapsed -ge 20){
                $remaining=($state.total-$state.processed)*$elapsed/$state.processed
                $text+=' | phase ETA ~{0:N0} min (finish ~{1})' -f ($remaining/60),(Get-Date).AddSeconds($remaining).ToString('HH:mm')
            }
        }
        Write-Host $text
        if($state.error){Write-Host $state.error}
        $last=$key
    }
    if($state.status -in @('FINISHED','FAILED','PAUSED')){break}
    Start-Sleep -Seconds $IntervalSeconds
}
