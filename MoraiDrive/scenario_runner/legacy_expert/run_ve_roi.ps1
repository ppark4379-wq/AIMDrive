param([double]$Seconds = 0, [switch]$MarkersOnly, [switch]$NoTrails)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$taskRun = Join-Path $taskRoot 'artifacts\ve_live'
New-Item -ItemType Directory -Force -Path $taskRun | Out-Null
$taskStop = Join-Path $taskRun 'roi.stop'
if (Test-Path -LiteralPath $taskStop) { Remove-Item -LiteralPath $taskStop }
$linuxRoot = (& wsl.exe -d Ubuntu-20.04 -- wslpath -a $taskRoot.Replace('\','/')).Trim()
$taskArgs = @('-d','Ubuntu-20.04','--','bash',"$linuxRoot/MoraiDrive/scenario_runner/legacy_expert/run_ve_roi.sh",'--seconds',"$Seconds",
              '--stop-file',"$linuxRoot/artifacts/ve_live/roi.stop")
if ($MarkersOnly) { $taskArgs += '--markers-only' }
if ($NoTrails) { $taskArgs += '--no-trails' }
& wsl.exe @taskArgs
