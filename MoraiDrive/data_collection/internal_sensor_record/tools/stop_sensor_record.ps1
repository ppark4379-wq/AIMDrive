# Fail-safe Sensor Record stop helper. It never clicks when the observed state
# is already OFF, and it never requests ON.
param([Parameter(Mandatory=$true)][int]$SimulatorProcessId,
      [Parameter(Mandatory=$true)][string]$OutputDirectory,
      [Parameter(Mandatory=$true)][string]$AllowedRoot,
      [int]$MaxAttempts=3)
$ErrorActionPreference='Stop'
function Test-AllowedOutputPath([string]$path) {
 $full=[IO.Path]::GetFullPath($path)
 foreach($relativeRoot in @('sensor_record_auto_v1','dataset\.collecting')) {
  $prefix=[IO.Path]::GetFullPath((Join-Path $AllowedRoot $relativeRoot)).TrimEnd('\')+'\'
  if ($full.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) {return $true}
 }
 return $false
}
if (-not (Test-AllowedOutputPath $OutputDirectory)) {throw 'Unexpected output root'}
if (Test-Path -LiteralPath $OutputDirectory) {throw 'Output already exists'}
if ($MaxAttempts -lt 1 -or $MaxAttempts -gt 5) {throw 'Use 1..5 attempts'}
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$env:TEMP=$OutputDirectory;$env:TMP=$OutputDirectory
Add-Type -AssemblyName System.Drawing
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class StopSensorRecordWindow {
 [StructLayout(LayoutKind.Sequential)] public struct Rect {public int Left,Top,Right,Bottom;}
 [StructLayout(LayoutKind.Sequential)] public struct Point {public int X,Y;}
 [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
 [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr w,int command);
 [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr w,out Rect r);
 [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr w,IntPtr h,uint f);
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
[void][StopSensorRecordWindow]::SetProcessDPIAware()
$sim=Get-Process -Id $SimulatorProcessId
if ($sim.ProcessName -ne 'Simulator') {throw 'Unexpected process'}
[void][StopSensorRecordWindow]::ShowWindow($sim.MainWindowHandle,9)
Start-Sleep -Milliseconds 700

function Observe([string]$name) {
 $r=New-Object StopSensorRecordWindow+Rect
 if (-not [StopSensorRecordWindow]::GetWindowRect($sim.MainWindowHandle,[ref]$r)) {throw 'No simulator window'}
 $width=$r.Right-$r.Left;$height=$r.Bottom-$r.Top
 if ($width -lt 800 -or $height -lt 600) {throw 'Simulator must remain restored'}
 $scale=[Math]::Min($width/1920.0,$height/1080.0)
 $bmp=New-Object Drawing.Bitmap($width,$height);$graphics=[Drawing.Graphics]::FromImage($bmp)
 try {
  $hdc=$graphics.GetHdc()
  try {$ok=[StopSensorRecordWindow]::PrintWindow($sim.MainWindowHandle,$hdc,2)} finally {$graphics.ReleaseHdc($hdc)}
  if (-not $ok) {throw 'Cannot read simulator UI'}
  $sync=$bmp.GetPixel([int][Math]::Round(25*$scale),[int][Math]::Round(130*$scale))
  $record=$bmp.GetPixel([int][Math]::Round(25*$scale),[int][Math]::Round(95*$scale))
  if ($sync.G -lt 150 -or $sync.R -gt 100 -or $sync.B -gt 100) {throw 'Expected visible internal synchronization panel'}
  $on=($record.R -gt 150 -and $record.G -lt 100 -and $record.B -lt 100)
  if (-not $on -and ([Math]::Abs([int]$record.R-[int]$record.G) -gt 12 -or $record.R -lt 20 -or $record.R -gt 110)) {
   throw 'Sensor Record indicator is not recognized'
  }
  if ($name) {$bmp.Save((Join-Path $OutputDirectory ($name+'.png')),[Drawing.Imaging.ImageFormat]::Png)}
  $observation=@{record_on=$on;width=$width;height=$height;scale=$scale;
   window_rect=@{left=$r.Left;top=$r.Top;right=$r.Right;bottom=$r.Bottom};
   foreground_handle=[StopSensorRecordWindow]::GetForegroundWindow().ToInt64();
   record_rgb=@($record.R,$record.G,$record.B);sync_rgb=@($sync.R,$sync.G,$sync.B);
   utc=[DateTime]::UtcNow.ToString('o');stopwatch_ticks=[Diagnostics.Stopwatch]::GetTimestamp()}
  $script:lastRecordObservation=$observation
  return $observation
 } finally {$graphics.Dispose();$bmp.Dispose()}
}

$events=New-Object Collections.Generic.List[object]
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
  $foreground=[StopSensorRecordWindow]::GetForegroundWindow();$foregroundPid=0
  $foregroundThread=[StopSensorRecordWindow]::GetWindowThreadProcessId($foreground,[ref]$foregroundPid)
  $currentThread=[StopSensorRecordWindow]::GetCurrentThreadId();$attached=$false
  try {
   if ($foregroundThread -ne 0 -and $foregroundThread -ne $currentThread) {
    $attached=[StopSensorRecordWindow]::AttachThreadInput($currentThread,$foregroundThread,$true)
   }
   [void][StopSensorRecordWindow]::BringWindowToTop($sim.MainWindowHandle)
   [void][StopSensorRecordWindow]::SetForegroundWindow($sim.MainWindowHandle)
   for($attempt=0;$attempt -lt 10 -and (Get-RecordClock) -lt $deadline -and
       [StopSensorRecordWindow]::GetForegroundWindow() -ne $sim.MainWindowHandle;$attempt++) {
    Start-Sleep -Milliseconds 100
   }
  } finally {
   if ($attached) {[void][StopSensorRecordWindow]::AttachThreadInput($currentThread,$foregroundThread,$false)}
  }
  if ([StopSensorRecordWindow]::GetForegroundWindow() -ne $sim.MainWindowHandle) {throw 'Simulator is not foreground; no click sent'}
  if ((Get-RecordClock) -ge $deadline) {throw 'Sensor Record transition deadline expired; no click sent'}
  $r=New-Object StopSensorRecordWindow+Rect
  if (-not [StopSensorRecordWindow]::GetWindowRect($sim.MainWindowHandle,[ref]$r)) {throw 'No simulator window; no click sent'}
  if ($r.Right-$r.Left -ne $state.width -or $r.Bottom-$r.Top -ne $state.height) {
   throw 'Simulator window changed since observation; no click sent'
  }
  $x=$r.Left+[int][Math]::Round(294*$state.scale)
  $y=$r.Top+[int][Math]::Round(95*$state.scale)
  if (-not [StopSensorRecordWindow]::SetCursorPos($x,$y)) {throw 'SetCursorPos failed; no click sent'}
  $cursor=New-Object StopSensorRecordWindow+Point
  if (-not [StopSensorRecordWindow]::GetCursorPos([ref]$cursor)) {throw 'GetCursorPos failed; no click sent'}
  $foreground=[StopSensorRecordWindow]::GetForegroundWindow()
  Add-RecordEvent 'click_prepared' @{observation=$state;
   window_rect=@{left=$r.Left;top=$r.Top;right=$r.Right;bottom=$r.Bottom};
   requested_cursor=@($x,$y);actual_cursor=@($cursor.X,$cursor.Y);
   foreground_handle=$foreground.ToInt64()}
  if ($cursor.X -ne $x -or $cursor.Y -ne $y) {throw 'Cursor did not reach Sensor Record switch; no click sent'}
  if ($foreground -ne $sim.MainWindowHandle) {throw 'Simulator lost foreground; no click sent'}
  [void][StopSensorRecordWindow]::mouse_event(2,0,0,0,[UIntPtr]::Zero)
  try {Start-Sleep -Milliseconds 80} finally {
   [void][StopSensorRecordWindow]::mouse_event(4,0,0,0,[UIntPtr]::Zero)
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

$errorText=$null;$state=$null
try {$state=Set-Record $false 'record_off' $MaxAttempts 12.}
catch {$errorText=$_.Exception.Message}
$result=[ordered]@{status=if ($errorText) {'off_failed'} else {'off_verified'};
 click_attempts=@($events | Where-Object {$_.event -eq 'click_sent'}).Count;
 observations=@($events | ForEach-Object {
  if ($_.observation) {$_.observation} elseif ($_.details.observation) {$_.details.observation}
 });events=@($events.ToArray());last_observation=$script:lastRecordObservation;
 record_on=if ($null -ne $script:lastRecordObservation) {$script:lastRecordObservation.record_on} else {$null};
 error=$errorText;transition_timeout_seconds=12.;training_visible=$false}
$result|ConvertTo-Json -Depth 10|Set-Content -LiteralPath (Join-Path $OutputDirectory 'stop_result.json') -Encoding UTF8
$result|ConvertTo-Json -Depth 10 -Compress
if ($errorText) {throw $errorText}
