# Windows regression: planning a new session must accept missing old targets.
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=New-Object Text.UTF8Encoding($false)
$fixture=Join-Path $env:TEMP ('morai-junction-test-'+[Guid]::NewGuid().ToString('N'))
$fixture=[IO.Path]::GetFullPath($fixture)
$tempRoot=[IO.Path]::GetFullPath($env:TEMP).TrimEnd('\')+'\'
if(!$fixture.StartsWith($tempRoot,[StringComparison]::OrdinalIgnoreCase)){
    throw 'Test path must be below TEMP'
}
$null=[IO.Directory]::CreateDirectory($fixture)
try {
    $storage=Join-Path $fixture 'storage'
    $sensors=Join-Path $fixture 'SensorData'
    $old=Join-Path $storage 'sensor_record_auto_v1\native_sessions\old_lap_003'
    $null=[IO.Directory]::CreateDirectory($sensors)
    $roles=@('CAMERA_1','CAMERA_2','CAMERA_3','CAMERA_4','LIDAR_5','GPS_6','IMU_7')
    foreach($role in $roles){
        $target=Join-Path $old $role
        $null=[IO.Directory]::CreateDirectory($target)
        New-Item -ItemType Junction -Path (Join-Path $sensors $role) -Target $target | Out-Null
        # Nonrecursive deletion of this test's empty target, leaving the junction.
        if(![IO.Path]::GetFullPath($target).StartsWith($fixture+'\',[StringComparison]::OrdinalIgnoreCase)){
            throw 'Test target escaped fixture'
        }
        [IO.Directory]::Delete($target)
    }
    [IO.Directory]::Delete($old)
    $worker=Join-Path $PSScriptRoot '../internal_sensor_record/tools/prepare_sensor_record_session.ps1'
    $result=& powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $worker -SessionName new_lap_001 -StorageRoot $storage -SensorDataRoot $sensors
    if($LASTEXITCODE -ne 0){throw 'Session planning failed'}
    $plan=($result -join [Environment]::NewLine) | ConvertFrom-Json
    if($plan.entries.Count -ne 7 -or $plan.bootstrap){throw 'Expected seven existing junctions'}
    if([IO.Directory]::Exists($plan.target_root)){throw 'Plan must not create a session'}
    if([IO.Directory]::Exists($old)){throw 'Plan must not recreate the old session'}
    foreach($role in $roles){
        $item=Get-Item -LiteralPath (Join-Path $sensors $role) -Force
        if($item.LinkType -ne 'Junction' -or $item.Target[0] -ne (Join-Path $old $role)){
            throw 'Plan changed a junction'
        }
    }
    Write-Host 'PASS: Windows plans new session from seven dangling junctions without changing them'
} finally {
    $resolved=[IO.Path]::GetFullPath($fixture)
    if(!$resolved.StartsWith($tempRoot,[StringComparison]::OrdinalIgnoreCase) -or
       [IO.Path]::GetFileName($resolved) -notlike 'morai-junction-test-*'){
        throw 'Unsafe fixture cleanup'
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
