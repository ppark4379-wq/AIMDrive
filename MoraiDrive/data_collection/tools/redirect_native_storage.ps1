# Legacy Sync Runtime storage redirect. Run in elevated Windows PowerShell only
# after stopping Simulator and collectors. Existing data is retained.
param([Parameter(Mandatory=$true)][string]$LauncherRoot,
      [Parameter(Mandatory=$true)][string]$RawRoot)
$ErrorActionPreference = 'Stop'
if ($LauncherRoot -notmatch '^[A-Za-z]:\\' -or
    $RawRoot -notmatch '^[A-Za-z]:\\') {
    throw 'LauncherRoot and RawRoot must be absolute Windows drive paths.'
}
$launcher = $LauncherRoot.TrimEnd('\')
$rawRoot = $RawRoot.TrimEnd('\')
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this script in Windows PowerShell as Administrator.'
}
if (Get-Process -Name Simulator -ErrorAction SilentlyContinue) {
    throw 'Close MORAI Simulator, Sync Runtime and Collector first.'
}
if (-not (Test-Path (Split-Path $rawRoot -Qualifier))) {
    throw 'RawRoot drive is unavailable in this administrator session.'
}
$pairs = @(
    @{ Source = "$launcher\SensorData\SynchronousMode"; Target = "$rawRoot\camera\SynchronousMode" },
    @{ Source = "$launcher\MoraiLauncher_Win_Data\SaveFile\SensorData\SynchronousMode"; Target = "$rawRoot\lidar\SynchronousMode" }
)
$changed = @()
try {
    foreach ($pair in $pairs) {
        New-Item -ItemType Directory -Path $pair.Target -Force | Out-Null
        $existing = Get-Item -LiteralPath $pair.Source -Force -ErrorAction SilentlyContinue
        if ($existing -and ($existing.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            if ($existing.LinkType -ne 'SymbolicLink' -or [string]$existing.Target -ne $pair.Target) {
                throw "Unexpected existing link: $($pair.Source)"
            }
            continue
        }
        $backup = $null
        if ($existing) {
            $backup = $pair.Source + '.before-redirect-' + [guid]::NewGuid().ToString('N')
            Move-Item -LiteralPath $pair.Source -Destination $backup
        }
        $change = @{ Source = $pair.Source; Backup = $backup; Linked = $false }
        $changed += $change
        New-Item -ItemType SymbolicLink -Path $pair.Source -Target $pair.Target | Out-Null
        $change.Linked = $true
    }
    foreach ($pair in $pairs) {
        $name = '.write-probe-' + [guid]::NewGuid().ToString('N')
        $probe = Join-Path $pair.Source $name
        $targetProbe = Join-Path $pair.Target $name
        try {
            [IO.File]::WriteAllText($probe, $name)
            if ([IO.File]::ReadAllText($targetProbe) -ne $name) { throw 'Redirection check failed.' }
        } finally {
            if (Test-Path -LiteralPath $probe) { Remove-Item -LiteralPath $probe }
        }
        Write-Host "Verified: $($pair.Source) -> $($pair.Target)"
    }
} catch {
    [array]::Reverse($changed)
    foreach ($change in $changed) {
        if ($change.Linked) { [IO.Directory]::Delete($change.Source) }
        if ($change.Backup) { Move-Item -LiteralPath $change.Backup -Destination $change.Source }
    }
    throw
}
foreach ($change in $changed) {
    if ($change.Backup) { Write-Host "Old data retained: $($change.Backup)" }
}
Write-Host 'Native storage redirected. Pass the corresponding WSL paths through collector CLI/environment; repository files were not edited.'
