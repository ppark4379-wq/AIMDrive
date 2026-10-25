$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$taskRun = Join-Path $taskRoot 'artifacts\ve_live'
New-Item -ItemType Directory -Force -Path $taskRun | Out-Null
[System.IO.File]::WriteAllText((Join-Path $taskRun 'roi.stop'), 'stop')
Write-Host 'Requested ROI viewer shutdown. Inference continues separately.'
