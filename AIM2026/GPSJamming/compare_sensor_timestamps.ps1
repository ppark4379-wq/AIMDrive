# Timestamp diagnostic only. Stop the live viewer before running.
$ErrorActionPreference = 'Stop'
& 'C:\conda-envs\botsort\python.exe' (Join-Path $PSScriptRoot 'scripts\compare_timestamps.py')
exit $LASTEXITCODE
