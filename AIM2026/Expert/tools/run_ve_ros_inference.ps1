param(
    [string]$Checkpoint = 'Y:\DCM\weights\checkpoint-best-object.pth.tar',
    [string]$Python = 'C:\Users\dydwn\anaconda3\envs\vldrive\python.exe',
    [double]$Seconds = 0
)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$taskRun = Join-Path $taskRoot 'artifacts\ve_live'
New-Item -ItemType Directory -Force -Path $taskRun | Out-Null
$taskStop = Join-Path $taskRun 'inference.stop'
$relayStop = Join-Path $taskRun 'relay.stop'
$taskMutex = New-Object System.Threading.Mutex($false, 'Local\AIMDriveVeRosInference')
$taskAcquired = $false
try { $taskAcquired = $taskMutex.WaitOne(0) }
catch [System.Threading.AbandonedMutexException] { $taskAcquired = $true }
if (-not $taskAcquired) {
    Write-Host '[VE] Inference is already running. You can open ROI with: bash tools/run_ve_roi.sh'
    Write-Host '[VE] To stop it from WSL: bash tools/run_ve_ros_inference.sh --stop'
    $taskMutex.Dispose()
    return
}
$relayProcess = $null
try {
    if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
    if (-not (Test-Path -LiteralPath $Checkpoint)) { throw "Checkpoint not found: $Checkpoint" }
    foreach ($stopPath in @($taskStop, $relayStop)) {
        if (Test-Path -LiteralPath $stopPath) { Remove-Item -LiteralPath $stopPath }
    }
    $linuxRoot = (& wsl.exe -d Ubuntu-20.04 -- wslpath -a $taskRoot.Replace('\','/')).Trim()
    $relayArgs = @('-d', 'Ubuntu-20.04', '--', 'bash', "$linuxRoot/AIM2026/Expert/tools/run_ve_ros_relay.sh",
                   '--stop-file', "$linuxRoot/artifacts/ve_live/relay.stop", '--report-dir', "$linuxRoot/artifacts/ve_live")
    $relayProcess = Start-Process -FilePath 'wsl.exe' -ArgumentList $relayArgs -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $taskRun 'relay.log') -RedirectStandardError (Join-Path $taskRun 'relay.err.log')
    Start-Sleep -Seconds 2
    if ($relayProcess.HasExited) { throw (Get-Content (Join-Path $taskRun 'relay.err.log') -Raw) }
    Write-Host 'VE perception only. No vehicle control. Ctrl+C stops inference and its relay.'
    & $Python -u (Join-Path $taskRoot 'VLDrive\inference\live_ros_windows.py') `
        --checkpoint $Checkpoint --seconds $Seconds --stop-file $taskStop
    $taskExit = $LASTEXITCODE
    if ($taskExit -ne 0) { throw "VE inference exited with code $taskExit" }
}
finally {
    [System.IO.File]::WriteAllText($relayStop, 'stop')
    if ($relayProcess -and -not $relayProcess.HasExited) { $relayProcess.WaitForExit(5000) | Out-Null }
    $taskMutex.ReleaseMutex()
    $taskMutex.Dispose()
}
