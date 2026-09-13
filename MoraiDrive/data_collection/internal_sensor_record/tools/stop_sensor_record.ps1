# Fail-safe Sensor Record stop helper. It never clicks when the observed state
# is already OFF, and it never requests ON.
param([Parameter(Mandatory=$true)][int]$SimulatorProcessId,
      [Parameter(Mandatory=$true)][string]$OutputDirectory,
      [Parameter(Mandatory=$true)][string]$AllowedRoot,
      [int]$MaxAttempts=3)
$ErrorActionPreference='Stop'
$allowedPrefix=(Join-Path $AllowedRoot 'sensor_record_auto_v1')+'\'
if (-not $OutputDirectory.StartsWith($allowedPrefix,[StringComparison]::OrdinalIgnoreCase)) {throw 'Unexpected output root'}
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
  $bmp.Save((Join-Path $OutputDirectory ($name+'.png')),[Drawing.Imaging.ImageFormat]::Png)
  return [ordered]@{record_on=$on;width=$width;height=$height;scale=$scale;
   record_rgb=@($record.R,$record.G,$record.B);sync_rgb=@($sync.R,$sync.G,$sync.B);
   utc=[DateTime]::UtcNow.ToString('o');stopwatch_ticks=[Diagnostics.Stopwatch]::GetTimestamp()}
 } finally {$graphics.Dispose();$bmp.Dispose()}
}

$observations=New-Object Collections.Generic.List[object]
for($attempt=0;$attempt -le $MaxAttempts;$attempt++) {
 $state=Observe ('observe_'+$attempt)
 $observations.Add($state)
 if (-not $state.record_on) {
  $result=[ordered]@{status='off_verified';click_attempts=$attempt;observations=@($observations.ToArray());training_visible=$false}
  $result|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $OutputDirectory 'stop_result.json') -Encoding UTF8
  $result|ConvertTo-Json -Depth 8 -Compress
  exit 0
 }
 if ($attempt -eq $MaxAttempts) {break}
 $r=New-Object StopSensorRecordWindow+Rect
 [void][StopSensorRecordWindow]::GetWindowRect($sim.MainWindowHandle,[ref]$r)
 $foreground=[StopSensorRecordWindow]::GetForegroundWindow();$foregroundPid=0
 $foregroundThread=[StopSensorRecordWindow]::GetWindowThreadProcessId($foreground,[ref]$foregroundPid)
 $currentThread=[StopSensorRecordWindow]::GetCurrentThreadId()
 $attached=$false
 try {
  if ($foregroundThread -ne 0 -and $foregroundThread -ne $currentThread) {
   $attached=[StopSensorRecordWindow]::AttachThreadInput($currentThread,$foregroundThread,$true)
  }
  [void][StopSensorRecordWindow]::BringWindowToTop($sim.MainWindowHandle)
  [void][StopSensorRecordWindow]::SetForegroundWindow($sim.MainWindowHandle)
  for($wait=0;$wait -lt 10 -and [StopSensorRecordWindow]::GetForegroundWindow() -ne $sim.MainWindowHandle;$wait++) {
   Start-Sleep -Milliseconds 100
  }
 } finally {
  if ($attached) {[void][StopSensorRecordWindow]::AttachThreadInput($currentThread,$foregroundThread,$false)}
 }
 if ([StopSensorRecordWindow]::GetForegroundWindow() -ne $sim.MainWindowHandle) {throw 'Simulator is not foreground; no click sent'}
 [void][StopSensorRecordWindow]::SetCursorPos($r.Left+[int][Math]::Round(294*$state.scale),$r.Top+[int][Math]::Round(95*$state.scale))
 [void][StopSensorRecordWindow]::mouse_event(2,0,0,0,[UIntPtr]::Zero)
 Start-Sleep -Milliseconds 100
 [void][StopSensorRecordWindow]::mouse_event(4,0,0,0,[UIntPtr]::Zero)
 Start-Sleep -Milliseconds 1500
}
throw 'Sensor Record remained ON after bounded stop attempts'
