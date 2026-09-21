# A bounded UI experiment. No sensor/network reconfiguration or vehicle commands.
# OutputDirectory must be a fresh directory under the configured local spool.
param([int]$SimulatorProcessId, [string]$OutputDirectory, [int]$WallSeconds=12,
      [Parameter(Mandatory=$true)][string]$AllowedRoot,
      [string]$CoordinationDirectory='', [switch]$ObserveOnly, [switch]$RestoreWindow)
$ErrorActionPreference='Stop'
if ($WallSeconds -lt 2 -or $WallSeconds -gt 3540) {throw 'Use 2..3540 wall seconds'}
function Test-AllowedCapturePath([string]$path) {
 $full=[IO.Path]::GetFullPath($path)
 foreach($relativeRoot in @('sensor_record_auto_v1','dataset\.collecting')) {
  $prefix=[IO.Path]::GetFullPath((Join-Path $AllowedRoot $relativeRoot)).TrimEnd('\')+'\'
  if ($full.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) {return $true}
 }
 return $false
}
if (-not (Test-AllowedCapturePath $OutputDirectory)) {throw 'Unexpected output root'}
if (Test-Path -LiteralPath $OutputDirectory) {throw 'Output already exists'}
if ($CoordinationDirectory) {
 if (-not (Test-AllowedCapturePath $CoordinationDirectory)) {
  throw 'Unexpected coordination root'
 }
 if (-not (Test-Path -LiteralPath $CoordinationDirectory -PathType Container)) {
  throw 'Coordination directory must be prepared before Sensor Record starts'
 }
}
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$env:TEMP=$OutputDirectory; $env:TMP=$OutputDirectory
Add-Type -AssemblyName System.Drawing
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class SensorRecordWindow {
 [StructLayout(LayoutKind.Sequential)] public struct Rect {public int Left,Top,Right,Bottom;}
 [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
 [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr w,int command);
 [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr w,out Rect r);
 [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr w,IntPtr h,uint f);
 [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr w,uint m,IntPtr a,IntPtr b);
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr w);
 [DllImport("user32.dll")] public static extern bool BringWindowToTop(IntPtr w);
 [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
 [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr w,out uint processId);
 [DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();
 [DllImport("user32.dll")] public static extern bool AttachThreadInput(uint from,uint to,bool attach);
 [DllImport("user32.dll")] public static extern bool SetCursorPos(int x,int y);
 [DllImport("user32.dll")] public static extern void mouse_event(uint f,uint x,uint y,uint d,UIntPtr e);
}
'@
[void][SensorRecordWindow]::SetProcessDPIAware()
$sim=Get-Process -Id $SimulatorProcessId
if ($sim.ProcessName -ne 'Simulator') {throw 'Unexpected process'}
if ($RestoreWindow) {
 [void][SensorRecordWindow]::ShowWindow($sim.MainWindowHandle,9)
 Start-Sleep -Milliseconds 700
}
$events=New-Object System.Collections.Generic.List[object]
function Join-Error([string]$primary,[string]$secondary) {
 if ([String]::IsNullOrWhiteSpace($primary)) {return $secondary}
 return ($primary+'; '+$secondary)
}
function Write-CoordinationMarker([string]$name,[hashtable]$payload) {
 if (-not $CoordinationDirectory) {return}
 if (-not (Test-AllowedCapturePath $CoordinationDirectory)) {
  throw 'Unexpected coordination root'
 }
 # The Python owner creates this before UI ownership. Recreate it defensively
 # if an external cleanup raced with shutdown, then publish the JSON atomically.
 if (-not (Test-Path -LiteralPath $CoordinationDirectory -PathType Container)) {
  New-Item -ItemType Directory -Path $CoordinationDirectory -Force | Out-Null
 }
 $target=Join-Path $CoordinationDirectory $name
 $temporary=Join-Path $CoordinationDirectory ($name+'.'+$PID+'.tmp')
 try {
  $payload|ConvertTo-Json -Compress|Set-Content -LiteralPath $temporary -Encoding UTF8
  Move-Item -LiteralPath $temporary -Destination $target -Force
 } finally {
  if (Test-Path -LiteralPath $temporary) {
   Remove-Item -LiteralPath $temporary -Force
  }
 }
}
function Observe([string]$name) {
 $r=New-Object SensorRecordWindow+Rect
 if (-not [SensorRecordWindow]::GetWindowRect($sim.MainWindowHandle,[ref]$r)) {throw 'No simulator window'}
 $width=$r.Right-$r.Left; $height=$r.Bottom-$r.Top
 if ($width -lt 800 -or $height -lt 600) {throw 'Simulator must remain restored'}
 # Unity's panel follows the smaller canvas scale on an ultrawide window.
 $s=[Math]::Min($width/1920.0,$height/1080.0)
 $bmp=New-Object System.Drawing.Bitmap($width,$height)
 $g=[System.Drawing.Graphics]::FromImage($bmp)
 try {
  $hdc=$g.GetHdc()
  try {$ok=[SensorRecordWindow]::PrintWindow($sim.MainWindowHandle,$hdc,2)} finally {$g.ReleaseHdc($hdc)}
  if (-not $ok) {throw 'Cannot read simulator UI'}
  $sync=$bmp.GetPixel([int][Math]::Round(25*$s),[int][Math]::Round(130*$s))
  $record=$bmp.GetPixel([int][Math]::Round(25*$s),[int][Math]::Round(95*$s))
  if ($sync.G -lt 150 -or $sync.R -gt 100 -or $sync.B -gt 100) {throw 'Expected visible internal synchronization panel'}
  # Sensor Record uses a RED lamp; Synchronization uses a GREEN lamp.
  $on=($record.R -gt 150 -and $record.G -lt 100 -and $record.B -lt 100)
  if (-not $on -and ([Math]::Abs([int]$record.R-[int]$record.G) -gt 12 -or $record.R -lt 20 -or $record.R -gt 110)) {
   throw 'Sensor Record indicator is not recognized'
  }
  if ($name) {$bmp.Save((Join-Path $OutputDirectory ($name+'.png')),[System.Drawing.Imaging.ImageFormat]::Png)}
  return @{record_on=$on;width=$width;height=$height;scale=$s;utc=[DateTime]::UtcNow.ToString('o');stopwatch_ticks=[Diagnostics.Stopwatch]::GetTimestamp()}
 } finally {$g.Dispose();$bmp.Dispose()}
}
function Set-Record([bool]$desired,[string]$name) {
 $state=Observe ''
 # A long Unity run can occasionally ignore a single foreground click. Retry
 # only after observing that the requested state was not reached; this avoids
 # blind double toggles.
 for($stateAttempt=0;$stateAttempt -lt 3 -and $state.record_on -ne $desired;$stateAttempt++) {
  $x=[int][Math]::Round(294*$state.scale);$y=[int][Math]::Round(95*$state.scale)
  $r=New-Object SensorRecordWindow+Rect
  [void][SensorRecordWindow]::GetWindowRect($sim.MainWindowHandle,[ref]$r)
  # SetForegroundWindow alone is advisory when the caller is launched through
  # WSL. AppActivate is executed immediately before the click, after scenario
  # setup has finished, then the real foreground handle remains the authority.
  $foreground=[SensorRecordWindow]::GetForegroundWindow();$foregroundPid=0
  $foregroundThread=[SensorRecordWindow]::GetWindowThreadProcessId($foreground,[ref]$foregroundPid)
  $currentThread=[SensorRecordWindow]::GetCurrentThreadId();$attached=$false
  try {
   if ($foregroundThread -ne 0 -and $foregroundThread -ne $currentThread) {
    $attached=[SensorRecordWindow]::AttachThreadInput($currentThread,$foregroundThread,$true)
   }
   [void][SensorRecordWindow]::BringWindowToTop($sim.MainWindowHandle)
   [void][SensorRecordWindow]::SetForegroundWindow($sim.MainWindowHandle)
   for($attempt=0;$attempt -lt 10 -and
       [SensorRecordWindow]::GetForegroundWindow() -ne $sim.MainWindowHandle;$attempt++) {
    Start-Sleep -Milliseconds 100
   }
  } finally {
   if ($attached) {[void][SensorRecordWindow]::AttachThreadInput($currentThread,$foregroundThread,$false)}
  }
  if ([SensorRecordWindow]::GetForegroundWindow() -ne $sim.MainWindowHandle) {throw 'Simulator is not foreground; no click sent'}
  [void][SensorRecordWindow]::SetCursorPos($r.Left+$x,$r.Top+$y)
  [void][SensorRecordWindow]::mouse_event(2,0,0,0,[UIntPtr]::Zero)
  Start-Sleep -Milliseconds 80
  [void][SensorRecordWindow]::mouse_event(4,0,0,0,[UIntPtr]::Zero)
  Start-Sleep -Milliseconds 1500
  $state=Observe ''
 }
 $after=Observe $name
 $events.Add(@{event=$name;observation=$after})
 if ($after.record_on -ne $desired) {throw 'UI state did not match requested Sensor Record state'}
 return $after
}
$errorText=$null; $attempted=$false
try {
 $before=Observe 'before'
 if ($ObserveOnly) {
  $events.Add(@{event='observe_only';observation=$before})
  return
 }
 if ($before.record_on) {throw 'Sensor Record was already ON; will not take ownership'}
 $events.Add(@{event='before';observation=$before})
 $attempted=$true
 $on=Set-Record $true 'record_on'
 Write-CoordinationMarker 'recording_armed.json' @{
  status='armed';utc=[DateTime]::UtcNow.ToString('o')
  stopwatch_ticks=[Diagnostics.Stopwatch]::GetTimestamp();record_on=$on.record_on}
 $timer=[Diagnostics.Stopwatch]::StartNew()
 while($timer.Elapsed.TotalSeconds -lt $WallSeconds) {
  if (Test-Path -LiteralPath (Join-Path $OutputDirectory 'stop.request')) {break}
  if ($CoordinationDirectory -and (Test-Path -LiteralPath (Join-Path $CoordinationDirectory 'stop.request'))) {break}
  Start-Sleep -Milliseconds 100
 }
} catch {$errorText=$_.Exception.Message} finally {
 if ($attempted) {
  try {$off=Set-Record $false 'record_off'}
  catch {$errorText=Join-Error $errorText ('OFF failed: '+$_.Exception.Message)}
 }
 if ($CoordinationDirectory -and $attempted) {
  try {
   Write-CoordinationMarker 'recording_stopped.json' @{
    status='stopped';utc=[DateTime]::UtcNow.ToString('o')
    stopwatch_ticks=[Diagnostics.Stopwatch]::GetTimestamp()
    record_on=if ($null -eq $off) {$null} else {$off.record_on}
    error=$errorText}
  } catch {
   $errorText=Join-Error $errorText (
    'recording_stopped acknowledgement failed: '+$_.Exception.Message)
  }
 }
 $report=@{events=@($events.ToArray());error=$errorText;wall_limit_seconds=$WallSeconds;
   stopwatch_frequency=[Diagnostics.Stopwatch]::Frequency;training_visible=$false}
 $report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $OutputDirectory 'ui_result.json') -Encoding UTF8
 $report|ConvertTo-Json -Depth 8 -Compress
}
if($errorText) {throw $errorText}
