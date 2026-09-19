# Plan by default. Preserve old data and junctions; change only the seven
# SensorData entries while recording is OFF.  A fresh workstation may still
# have ordinary directories there; those directories are renamed intact
# before the first junctions are created.
param([Parameter(Mandatory=$true)][string]$SessionName,
      [Parameter(Mandatory=$true)][string]$StorageRoot,
      [Parameter(Mandatory=$true)][string]$SensorDataRoot,
      [string]$TargetRoot='',
      [string]$ExpectedVolumeId='', [Int64]$ReserveBytes=53687091200,
      [switch]$Apply, [string]$PlanPath, [int]$SimulatorProcessId)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false)
if ($SessionName -notmatch '^[A-Za-z0-9_]{1,80}$') {throw 'Invalid session name'}
if ($StorageRoot -notmatch '^[A-Za-z]:\\' -or $StorageRoot.Contains("'")) {throw 'StorageRoot must be a local Windows drive path'}
if ($SensorDataRoot -notmatch '^[A-Za-z]:\\' -or $SensorDataRoot.Contains("'")) {throw 'SensorDataRoot must be an absolute Windows drive path'}
$sourceRoot=$SensorDataRoot.TrimEnd('\\')
$base=Join-Path $StorageRoot 'sensor_record_auto_v1'
$datasetRoot=Join-Path $StorageRoot 'dataset'
$collectingRoot=Join-Path $datasetRoot '.collecting'
if ($TargetRoot) {
 $targetRoot=[IO.Path]::GetFullPath($TargetRoot).TrimEnd('\\')
 $collectingPrefix=[IO.Path]::GetFullPath($collectingRoot).TrimEnd('\\')+'\\'
 if (-not $targetRoot.StartsWith($collectingPrefix,[StringComparison]::OrdinalIgnoreCase) -or
     (Split-Path $targetRoot -Leaf) -cne $SessionName) {
  throw 'TargetRoot must be this session below dataset\\.collecting'
 }
 $relative=$targetRoot.Substring($collectingPrefix.Length)
 if (@($relative.Split('\\',[StringSplitOptions]::RemoveEmptyEntries)).Count -ne 3) {
  throw 'TargetRoot must be dataset\\.collecting\\scenario\\campaign\\run'
 }
} else {
 $targetRoot=Join-Path (Join-Path $base 'native_sessions') $SessionName
}
$roles=@('CAMERA_1','CAMERA_2','CAMERA_3','CAMERA_4','LIDAR_5','GPS_6','IMU_7')
$driveLetter=$StorageRoot.Substring(0,1).ToUpperInvariant()
$volume=Get-Volume -DriveLetter $driveLetter
if (($ExpectedVolumeId -and $volume.UniqueId -ne $ExpectedVolumeId) -or
    [string]$volume.HealthStatus -ne 'Healthy' -or $volume.SizeRemaining -lt $ReserveBytes) {
 throw 'Wrong/unhealthy/full acquisition volume; no changes made'
}
if (Test-Path -LiteralPath $targetRoot) {throw 'Session target already exists; never reuse it'}
$entries=@();$oldRoots=@();$oldKinds=@()
foreach ($role in $roles) {
 $source=Join-Path $sourceRoot $role
 $item=Get-Item -LiteralPath $source -Force
 $targets=@($item.Target);$old=$null;$oldRoot=$null
 if ($item.LinkType -eq 'Junction' -and $targets.Count -eq 1) {
  $kind='Junction';$old=$targets[0];$oldRoot=Split-Path $old -Parent
  $nativeSessions=Join-Path $base 'native_sessions'
  $oldSession=Split-Path $oldRoot -Leaf
  $oldSessionParent=Split-Path $oldRoot -Parent
  $oldFull=[IO.Path]::GetFullPath($oldRoot).TrimEnd('\\')
  $collectingPrefix=[IO.Path]::GetFullPath($collectingRoot).TrimEnd('\\')+'\\'
  $approvedDirect=$oldFull.StartsWith($collectingPrefix,[StringComparison]::OrdinalIgnoreCase)
  $approvedOld=(($oldRoot -eq (Join-Path $base 'native')) -or
      ($oldSessionParent -eq $nativeSessions -and $oldSession -match '^[A-Za-z0-9_]{1,80}$') -or
      $approvedDirect)
  if ((Split-Path $old -Leaf) -cne $role -or -not $approvedOld) {
   throw 'Unexpected previous target; no redirect'
  }
  $backup=$source+'.link_before_'+$SessionName
  $oldRoots+= $oldRoot
 } elseif ($item.PSIsContainer -and -not $item.LinkType) {
  $kind='Directory'
  $backup=$source+'.data_before_'+$SessionName
 } else {
  throw 'SensorData entry must be one approved junction or one ordinary directory'
 }
 if (Test-Path -LiteralPath $backup) {throw 'Preserved SensorData path exists'}
 $oldKinds+= $kind
 $entries+= [ordered]@{role=$role;source=$source;old_kind=$kind;old_target=$old;
                       preserved_source=$backup;target=(Join-Path $targetRoot $role)}
}
if (@($oldKinds | Select-Object -Unique).Count -ne 1) {throw 'SensorData entry kinds are mixed'}
$bootstrap=($oldKinds[0] -eq 'Directory')
if (-not $bootstrap -and @($oldRoots | Select-Object -Unique).Count -ne 1) {throw 'Existing junction set is mixed'}
$oldRoot=$null
if (-not $bootstrap) {$oldRoot=$oldRoots[0]}
$plan=[ordered]@{schema='morai_sensor_record_native_session_v2';session=$SessionName;
 source_root=$sourceRoot;target_root=$targetRoot;old_root=$oldRoot;bootstrap=$bootstrap;
 entries=$entries;source_directories_preserved=$true;source_files_deleted=$false}
if (-not $Apply) {$plan | ConvertTo-Json -Depth 8 -Compress;exit 0}
if (-not $PlanPath -or -not $SimulatorProcessId) {throw 'Apply requires a saved plan and simulator PID'}
$expected=Get-Content -LiteralPath $PlanPath -Raw | ConvertFrom-Json
if ($expected.schema -ne $plan.schema -or $expected.session -cne $SessionName -or
    $expected.target_root -cne $targetRoot -or @($expected.entries).Count -ne 7) {throw 'Plan identity differs'}
for ($i=0;$i -lt 7;$i++) {
 foreach ($field in @('role','source','old_kind','old_target','preserved_source','target')) {
  if ($expected.entries[$i].$field -cne $entries[$i][$field]) {throw 'Plan changed; no mutations made'}
 }
}
$env:TEMP=Split-Path $PlanPath -Parent; $env:TMP=$env:TEMP
Add-Type -AssemblyName System.Drawing
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class NativeSessionWindow {
 [StructLayout(LayoutKind.Sequential)] public struct Rect {public int Left,Top,Right,Bottom;}
 [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
 [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr w,int command);
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr w);
 [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr w,out Rect r);
 [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr w,IntPtr h,uint f);
}
'@
[void][NativeSessionWindow]::SetProcessDPIAware()
function Assert-RecordOff {
 $sim=Get-Process -Id $SimulatorProcessId
 if ($sim.ProcessName -ne 'Simulator') {throw 'Simulator identity changed'}
 $r=New-Object NativeSessionWindow+Rect
 if (-not [NativeSessionWindow]::GetWindowRect($sim.MainWindowHandle,[ref]$r)) {throw 'No window'}
 $w=$r.Right-$r.Left;$h=$r.Bottom-$r.Top
 if ($w -lt 800 -or $h -lt 600) {
  [void][NativeSessionWindow]::ShowWindow($sim.MainWindowHandle,9)
  [void][NativeSessionWindow]::SetForegroundWindow($sim.MainWindowHandle)
  Start-Sleep -Milliseconds 700
  if (-not [NativeSessionWindow]::GetWindowRect($sim.MainWindowHandle,[ref]$r)) {throw 'No window after restore'}
  $w=$r.Right-$r.Left;$h=$r.Bottom-$r.Top
  if ($w -lt 800 -or $h -lt 600) {throw 'Cannot restore simulator for recording-state check'}
 }
 $bmp=New-Object System.Drawing.Bitmap($w,$h);$g=[Drawing.Graphics]::FromImage($bmp)
 try {
  $hdc=$g.GetHdc()
  try {$ok=[NativeSessionWindow]::PrintWindow($sim.MainWindowHandle,$hdc,2)} finally {$g.ReleaseHdc($hdc)}
  if (-not $ok) {throw 'Cannot observe recording state'}
  $s=[Math]::Min($w/1920.0,$h/1080.0);$record=$bmp.GetPixel([int][Math]::Round(25*$s),[int][Math]::Round(95*$s))
  $sync=$bmp.GetPixel([int][Math]::Round(25*$s),[int][Math]::Round(130*$s))
  if ($sync.G -lt 150 -or $sync.R -gt 100 -or $sync.B -gt 100 -or
      [Math]::Abs([int]$record.R-[int]$record.G) -gt 12 -or
      [Math]::Abs([int]$record.R-[int]$record.B) -gt 12 -or $record.R -lt 20 -or $record.R -gt 110) {
   $bmp.Save((Join-Path (Split-Path $PlanPath -Parent) 'session_off_check_failed.png'),[Drawing.Imaging.ImageFormat]::Png)
   throw ('Sensor Record OFF / sync ON unconfirmed; record='+$record.ToString()+' sync='+$sync.ToString()+' window='+$w+'x'+$h)
  }
 } finally {$g.Dispose();$bmp.Dispose()}
}
Assert-RecordOff
# Any partial failure is left visible; the Python preflight rejects mixed roots.
New-Item -ItemType Directory -Path $targetRoot | Out-Null
foreach ($row in $entries) {New-Item -ItemType Directory -Path $row.target | Out-Null}
foreach ($row in $entries) {
 Assert-RecordOff
 $current=Get-Item -LiteralPath $row.source -Force
 if ($row.old_kind -eq 'Junction') {
  if ($current.LinkType -ne 'Junction' -or @($current.Target).Count -ne 1 -or
      $current.Target[0] -cne $row.old_target) {throw 'Junction changed during preparation'}
 } elseif (-not $current.PSIsContainer -or $current.LinkType) {
  throw 'SensorData directory changed during preparation'
 }
 Rename-Item -LiteralPath $row.source -NewName (Split-Path $row.preserved_source -Leaf)
 New-Item -ItemType Junction -Path $row.source -Target $row.target | Out-Null
 $new=Get-Item -LiteralPath $row.source -Force
 $old=Get-Item -LiteralPath $row.preserved_source -Force
 if ($new.LinkType -ne 'Junction' -or $new.Target[0] -cne $row.target) {throw 'Redirect verification failed'}
 if ($row.old_kind -eq 'Junction') {
  if ($old.LinkType -ne 'Junction' -or $old.Target[0] -cne $row.old_target) {throw 'Preserved junction verification failed'}
 } elseif (-not $old.PSIsContainer -or $old.LinkType) {
  throw 'Preserved directory verification failed'
 }
}
Assert-RecordOff
$record=[ordered]@{applied=$true;plan=$plan;utc=[DateTime]::UtcNow.ToString('o');
 simulator_pid=$SimulatorProcessId;training_visible=$false}
$stream=[IO.File]::Open((Join-Path $targetRoot 'SESSION.json'),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try {
 $bytes=[Text.Encoding]::UTF8.GetBytes(($record | ConvertTo-Json -Depth 10 -Compress))
 $stream.Write($bytes,0,$bytes.Length);$stream.Flush($true)
} finally {$stream.Dispose()}
$record | ConvertTo-Json -Depth 10 -Compress
