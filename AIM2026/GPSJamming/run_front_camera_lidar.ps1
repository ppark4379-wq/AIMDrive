# Front camera (5050): YOLOv7 -> BoT-SORT -> LiDAR projection/association viewer.
$ErrorActionPreference = 'Stop'

$pythonExe = 'C:\conda-envs\botsort\python.exe'
$weightsPath = Join-Path $env:USERPROFILE 'Projects\BoT-SORT\pretrained\yolov7.pt'
$viewerPath = Join-Path $PSScriptRoot 'scripts\view_association.py'

foreach ($requiredPath in @($pythonExe, $weightsPath, $viewerPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required file not found: $requiredPath"
    }
}

Write-Host 'Front camera :5050 + LiDAR :2369 | YOLOv7 + BoT-SORT + association'
Write-Host 'Q / ESC: quit   S: save preview'

$viewerArgs = @(
    $viewerPath,
    '--detector', 'yolov7',
    '--weights', $weightsPath,
    '--port', '5050',
    '--device', '0',
    '--fp16',
    '--img-size', '640'
)
& $pythonExe @viewerArgs
exit $LASTEXITCODE
