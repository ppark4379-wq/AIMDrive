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
 [StructLayout(LayoutKind.Sequential)] public struct Point {public int X,Y;}
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
 [DllImport("user32.dll")] public static extern bool GetCursorPos(out Point p);
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
 if ($width -lt 800 -or $height -lt 600) {
  # The simulator may be minimized while a run is recording. Restore before
  # reading the indicator so the bounded OFF transition can still complete.
  [void][SensorRecordWindow]::ShowWindow($sim.MainWindowHandle,9)
  Start-Sleep -Milliseconds 700
  if (-not [SensorRecordWindow]::GetWindowRect($sim.MainWindowHandle,[ref]$r)) {throw 'No simulator window'}
  $width=$r.Right-$r.Left; $height=$r.Bottom-$r.Top
  if ($width -lt 800 -or $height -lt 600) {throw 'Simulator must remain restored'}
 }
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
  $observation=@{record_on=$on;width=$width;height=$height;scale=$s;
   window_rect=@{left=$r.Left;top=$r.Top;right=$r.Right;bottom=$r.Bottom};
   record_rgb=@($record.R,$record.G,$record.B);sync_rgb=@($sync.R,$sync.G,$sync.B);
   foreground_handle=[SensorRecordWindow]::GetForegroundWindow().ToInt64();
   utc=[DateTime]::UtcNow.ToString('o');stopwatch_ticks=[Diagnostics.Stopwatch]::GetTimestamp()}
  $script:lastRecordObservation=$observation
  return $observation
 } finally {$g.Dispose();$bmp.Dispose()}
}
function Get-RecordClock {
 return ([Diagnostics.Stopwatch]::GetTimestamp()/[double][Diagnostics.Stopwatch]::Frequency)
}
function Add-RecordEvent([string]$name,[hashtable]$details) {
 $events.Add(@{event=$name;utc=[DateTime]::UtcNow.ToString('o');
   stopwatch_ticks=[Diagnostics.Stopwatch]::GetTimestamp();details=$details})
}
function Invoke-RecordClick([hashtable]$state,[double]$deadline) {
  # SetForegroundWindow alone is advisory when the caller is launched through
  # WSL. The real foreground handle remains the authority before any click.
  $foreground=[SensorRecordWindow]::GetForegroundWindow();$foregroundPid=0
  $foregroundThread=[SensorRecordWindow]::GetWindowThreadProcessId($foreground,[ref]$foregroundPid)
  $currentThread=[SensorRecordWindow]::GetCurrentThreadId();$attached=$false
  try {
   if ($foregroundThread -ne 0 -and $foregroundThread -ne $currentThread) {
    $attached=[SensorRecordWindow]::AttachThreadInput($currentThread,$foregroundThread,$true)
   }
   [void][SensorRecordWindow]::BringWindowToTop($sim.MainWindowHandle)
   [void][SensorRecordWindow]::SetForegroundWindow($sim.MainWindowHandle)
   for($attempt=0;$attempt -lt 10 -and (Get-RecordClock) -lt $deadline -and
       [SensorRecordWindow]::GetForegroundWindow() -ne $sim.MainWindowHandle;$attempt++) {
    Start-Sleep -Milliseconds 100
   }
  } finally {
   if ($attached) {[void][SensorRecordWindow]::AttachThreadInput($currentThread,$foregroundThread,$false)}
  }
  if ([SensorRecordWindow]::GetForegroundWindow() -ne $sim.MainWindowHandle) {throw 'Simulator is not foreground; no click sent'}
  if ((Get-RecordClock) -ge $deadline) {throw 'Sensor Record transition deadline expired; no click sent'}
  $r=New-Object SensorRecordWindow+Rect
  if (-not [SensorRecordWindow]::GetWindowRect($sim.MainWindowHandle,[ref]$r)) {throw 'No simulator window; no click sent'}
  if ($r.Right-$r.Left -ne $state.width -or $r.Bottom-$r.Top -ne $state.height) {
   throw 'Simulator window changed since observation; no click sent'
  }
  $x=$r.Left+[int][Math]::Round(294*$state.scale)
  $y=$r.Top+[int][Math]::Round(95*$state.scale)
  if (-not [SensorRecordWindow]::SetCursorPos($x,$y)) {throw 'SetCursorPos failed; no click sent'}
  $cursor=New-Object SensorRecordWindow+Point
  if (-not [SensorRecordWindow]::GetCursorPos([ref]$cursor)) {throw 'GetCursorPos failed; no click sent'}
  $foreground=[SensorRecordWindow]::GetForegroundWindow()
  Add-RecordEvent 'click_prepared' @{observation=$state;
   window_rect=@{left=$r.Left;top=$r.Top;right=$r.Right;bottom=$r.Bottom};
   requested_cursor=@($x,$y);actual_cursor=@($cursor.X,$cursor.Y);
   foreground_handle=$foreground.ToInt64()}
  if ($cursor.X -ne $x -or $cursor.Y -ne $y) {throw 'Cursor did not reach Sensor Record switch; no click sent'}
  if ($foreground -ne $sim.MainWindowHandle) {throw 'Simulator lost foreground; no click sent'}
  [void][SensorRecordWindow]::mouse_event(2,0,0,0,[UIntPtr]::Zero)
  try {Start-Sleep -Milliseconds 80} finally {
   [void][SensorRecordWindow]::mouse_event(4,0,0,0,[UIntPtr]::Zero)
  }
  Add-RecordEvent 'click_sent' @{requested_cursor=@($x,$y);actual_cursor=@($cursor.X,$cursor.Y)}
}
function Set-Record([bool]$desired,[string]$name,[int]$MaxAttempts=3,[double]$TimeoutSeconds=12.) {
 if ($MaxAttempts -lt 1 -or $MaxAttempts -gt 5 -or $TimeoutSeconds -le 0. -or $TimeoutSeconds -gt 12.) {
  throw 'Use 1..5 attempts and a transition budget up to 12 seconds'
 }
 # The entire transition, including foreground retries, shares one budget.
 # Poll after each click before considering another toggle; OFF never clicks
 # again after a recognized OFF observation.
 $deadline=(Get-RecordClock)+$TimeoutSeconds
 try {
  $state=Observe ''
  Add-RecordEvent 'transition_begin' @{desired=$desired;timeout_seconds=$TimeoutSeconds;observation=$state}
  for($stateAttempt=0;$stateAttempt -lt $MaxAttempts -and $state.record_on -ne $desired;$stateAttempt++) {
   if ((Get-RecordClock) -ge $deadline) {break}
   Invoke-RecordClick $state $deadline
   $pollDeadline=[Math]::Min($deadline,(Get-RecordClock)+4.)
   while ((Get-RecordClock) -lt $pollDeadline) {
    Start-Sleep -Milliseconds ([int][Math]::Min(150.,[Math]::Max(1.,1000.*($pollDeadline-(Get-RecordClock)))))
    $state=Observe ''
    Add-RecordEvent 'transition_observed' @{attempt=($stateAttempt+1);desired=$desired;observation=$state}
    if ($state.record_on -eq $desired) {break}
   }
  }
  $after=Observe $name
  $events.Add(@{event=$name;observation=$after})
  if ($after.record_on -ne $desired) {throw 'UI state did not match requested Sensor Record state within bounded transition'}
  return $after
 } catch {
  Add-RecordEvent 'transition_failed' @{desired=$desired;error=$_.Exception.Message;last_observation=$script:lastRecordObservation}
  throw
 }
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
    record_on=if ($null -ne $script:lastRecordObservation) {$script:lastRecordObservation.record_on} else {$null}
    last_observation=$script:lastRecordObservation
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
