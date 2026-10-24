$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$taskRun = Join-Path $taskRoot 'artifacts\ve_live'
New-Item -ItemType Directory -Force -Path $taskRun | Out-Null
[System.IO.File]::WriteAllText((Join-Path $taskRun 'inference.stop'), 'stop')
Write-Host 'Requested VE inference shutdown. MORAI and rosbridge remain running.'
