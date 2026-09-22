$ErrorActionPreference='Stop'
$fixture=Join-Path $env:TEMP ('morai-watchdog-test-'+[Guid]::NewGuid().ToString('N'))
$null=[IO.Directory]::CreateDirectory($fixture)
try {
    $worker=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../internal_sensor_record/tools/publish_sensor_record_nas.ps1'))
    $child=Join-Path $fixture 'child.ps1'
    $script=@'
param([string]$Worker,[string]$State,[string]$Mode)
. $Worker
Initialize-NativeWatchdog
$lease=Join-Path $State 'parent.lease'
[IO.File]::WriteAllText($lease,'active')
[MoraiNasWatchdog]::Watch($lease)
[MoraiNasWatchdog]::Progress('copying','CAMERA_1/test.png',10,3,1,2,0)
[MoraiNasWatchdog]::Begin('directory_exists','Z:\DCM')
if($Mode -eq 'completed'){
    [MoraiNasWatchdog]::Begin('rename_copied_file','Z:\DCM\completed.txt')
    [Threading.Thread]::Sleep(20)
    [MoraiNasWatchdog]::End('')
}
$field=[MoraiNasWatchdog].GetField('pulse',[Reflection.BindingFlags]'NonPublic,Static')
$field.SetValue($null,[long][DateTime]::UtcNow.AddSeconds(-121).Ticks)
[Threading.Thread]::Sleep(10000)
throw 'Watchdog failed to exit'
'@
    [IO.File]::WriteAllText($child,$script,[Text.UTF8Encoding]::new($false))
    foreach($mode in @('blocked','completed')) {
        $state=Join-Path $fixture $mode
        $null=[IO.Directory]::CreateDirectory($state)
        $args=@('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',('"'+$child+'"'),'-Worker',('"'+$worker+'"'),'-State',('"'+$state+'"'),'-Mode',$mode)
        $process=Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList $args -WindowStyle Hidden -PassThru
        if(-not $process.WaitForExit(15000)){Stop-Process -Id $process.Id -Force;throw 'Watchdog test child timed out'}
        $exitCode=$process.ExitCode
        if($null -eq $exitCode){$process.Refresh();$exitCode=$process.ExitCode}
        $failure=Get-Content -LiteralPath (Join-Path $state 'watchdog_failure.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        if($failure.reason -ne 'no_progress_over_120_seconds' -or $failure.processed_files -ne 3 -or $failure.pulse_age_ms -lt 120000){
            throw 'Watchdog cause or counters were not preserved'
        }
        if($mode -eq 'blocked' -and ($failure.operation -ne 'directory_exists' -or $failure.path -ne 'Z:\DCM')){
            throw 'Blocked operation was not preserved'
        }
        if($mode -eq 'completed' -and ($failure.operation -ne 'idle' -or
            $failure.last_completed_operation -ne 'rename_copied_file' -or
            $failure.last_completed_path -ne 'Z:\DCM\completed.txt' -or $failure.last_completed_elapsed_ms -lt 15)){
            throw 'Operation completed immediately before watchdog exit was not preserved'
        }
        if($null -ne $exitCode -and $exitCode -ne 4){throw "Unexpected watchdog exit $exitCode"}
        Write-Host ("PASS: watchdog preserves {0} operation at exit" -f $mode)
    }
} finally {
    $resolved=[IO.Path]::GetFullPath($fixture)
    $tempRoot=[IO.Path]::GetFullPath($env:TEMP).TrimEnd('\')+'\'
    if(!$resolved.StartsWith($tempRoot,[StringComparison]::OrdinalIgnoreCase) -or
       [IO.Path]::GetFileName($resolved) -notlike 'morai-watchdog-test-*'){throw 'Unsafe test cleanup'}
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
