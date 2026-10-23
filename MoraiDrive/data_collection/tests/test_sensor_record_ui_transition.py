"""Run the real PS transition helpers with fake Win32, clocks and observations.

Only function ASTs are evaluated. Neither script's initialization, P/Invoke,
simulator process lookup, screen capture nor filesystem/UI actions can run.
"""
import base64
import json
from pathlib import Path
import shutil
import subprocess
import unittest


TOOLS = Path(__file__).resolve().parents[1] / "internal_sensor_record/tools"
POWERSHELL = shutil.which("powershell.exe") or shutil.which("pwsh")

HARNESS = r"""
$ErrorActionPreference='Stop'
Add-Type -TypeDefinition @'
using System;
public static class HarnessWindow {
 public struct Rect {public int Left,Top,Right,Bottom;}
 public struct Point {public int X,Y;}
 public static bool CursorSet=true,CursorRead=true,WrongCursor=false,Foreground=true,LostAfterCursor=false;
 public static int Width=1931,CursorX=0,CursorY=0,Down=0,Up=0;
 public static IntPtr GetForegroundWindow(){return new IntPtr(Foreground?123:124);}
 public static uint GetWindowThreadProcessId(IntPtr w,out uint p){p=99;return 1;}
 public static uint GetCurrentThreadId(){return 1;}
 public static bool AttachThreadInput(uint a,uint b,bool c){return true;}
 public static bool BringWindowToTop(IntPtr w){return true;}
 public static bool SetForegroundWindow(IntPtr w){return true;}
 public static bool GetWindowRect(IntPtr w,out Rect r){r=new Rect{Left=20,Top=30,Right=20+Width,Bottom=1014};return true;}
 public static bool SetCursorPos(int x,int y){CursorX=x;CursorY=y;if(LostAfterCursor)Foreground=false;return CursorSet;}
 public static bool GetCursorPos(out Point p){p=new Point{X=CursorX+(WrongCursor?1:0),Y=CursorY};return CursorRead;}
 public static void mouse_event(uint f,uint x,uint y,uint d,UIntPtr e){if(f==2)Down++;if(f==4)Up++;}
}
'@
function Assert([bool]$condition,[string]$message){if(-not $condition){throw $message}}
function Get-RecordClock {return $script:clock}
function Start-Sleep([int]$Milliseconds){$script:clock += $Milliseconds/1000.}
function Observe([string]$name){
 if($script:case -eq 'observe_failure' -and [HarnessWindow]::Down -gt 0){throw 'Fake pixel read failure'}
 $on=$script:case -ne 'already_off'
 if($script:case -eq 'delayed_off' -and $script:clock -ge 2.8){$on=$false}
 $observation=@{record_on=$on;width=1931;height=984;scale=(984/1080.);
   record_rgb=if($on){@(255,0,0)}else{@(64,64,66)};sync_rgb=@(0,255,0);
   utc='mock:'+ $script:clock;stopwatch_ticks=[int]($script:clock*1000.)}
 $script:lastRecordObservation=$observation
 return $observation
}
function Reset([string]$name){
 $script:case=$name;$script:clock=0.;$script:lastRecordObservation=$null
 $script:events=New-Object Collections.Generic.List[object]
 [HarnessWindow]::CursorSet=$true;[HarnessWindow]::CursorRead=$true;[HarnessWindow]::WrongCursor=$false
 [HarnessWindow]::Foreground=$true;[HarnessWindow]::LostAfterCursor=$false
 [HarnessWindow]::Width=1931;[HarnessWindow]::Down=0;[HarnessWindow]::Up=0
}
$results=New-Object Collections.Generic.List[object]
foreach($sourceRow in $sources){
 $source=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($sourceRow.source))
 $tokens=$null;$errors=$null
 $ast=[Management.Automation.Language.Parser]::ParseInput($source,[ref]$tokens,[ref]$errors)
 Assert ($errors.Count -eq 0) ('Parser error in '+$sourceRow.name+': '+($errors|Out-String))
 foreach($name in @('Add-RecordEvent','Invoke-RecordClick','Set-Record')){
  $function=$ast.Find({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name},$true)
  Assert ($null -ne $function) ('Missing helper '+$name)
  $safe=$function.Extent.Text.Replace('StopSensorRecordWindow','HarnessWindow').Replace('SensorRecordWindow','HarnessWindow')
  Invoke-Expression $safe
 }
 $sim=@{MainWindowHandle=[IntPtr]123}
 foreach($name in @('already_off','delayed_off','persistent_on','cursor_set_failure','cursor_read_failure','wrong_cursor','lost_foreground','changed_window','observe_failure','foreground_timeout')){
  Reset $name
  switch($name){
   'cursor_set_failure' {[HarnessWindow]::CursorSet=$false}
   'cursor_read_failure' {[HarnessWindow]::CursorRead=$false}
   'wrong_cursor' {[HarnessWindow]::WrongCursor=$true}
   'lost_foreground' {[HarnessWindow]::LostAfterCursor=$true}
   'changed_window' {[HarnessWindow]::Width=1950}
   'foreground_timeout' {[HarnessWindow]::Foreground=$false}
  }
  $harnessError=$null;$result=$null
  try{$result=Set-Record $false 'record_off' 3 12.}catch{$harnessError=$_.Exception.Message}
  Assert ([HarnessWindow]::Down -eq [HarnessWindow]::Up) ('Mouse button not released: '+$name)
  Assert ($script:clock -le 12.01) ('Transition exceeded total budget: '+$name)
  if($name -in @('already_off','delayed_off')){
   Assert ($null -eq $harnessError -and $result.record_on -eq $false) ('Expected verified OFF: '+$name)
   $expected=if($name -eq 'already_off'){0}else{1}
   Assert ([HarnessWindow]::Down -eq $expected) ('Unexpected extra toggle: '+$name)
  }else{
   Assert ($null -ne $harnessError -and $script:lastRecordObservation.record_on -eq $true) ('Lost actual ON on failure: '+$name)
   if($name -eq 'persistent_on'){
    Assert ([HarnessWindow]::Down -eq 3 -and $script:clock -ge 11.9) 'Persistent ON must use bounded polling, not rapid toggles'
   }elseif($name -eq 'observe_failure'){
    Assert ([HarnessWindow]::Down -eq 1) 'Pixel read failure must not trigger blind retry'
   }else{
    Assert ([HarnessWindow]::Down -eq 0) ('Unsafe input sent: '+$name)
   }
  }
  if($name -eq 'delayed_off'){
   $click=@($events|Where-Object {$_.event -eq 'click_prepared'})[0].details
   Assert ($click.requested_cursor[0] -eq 288 -and $click.requested_cursor[1] -eq 117) 'Window-relative click coordinates wrong'
   Assert ($click.actual_cursor[0] -eq 288 -and $click.actual_cursor[1] -eq 117) 'Actual cursor evidence missing'
   Assert ($click.foreground_handle -eq 123 -and $click.observation.record_rgb[0] -eq 255) 'Click evidence missing foreground/pixels'
   Assert (@($events|Where-Object {$_.event -eq 'transition_observed'}).Count -gt 1) 'Delayed OFF was not polled'
  }
  $results.Add(@{tool=$sourceRow.name;case=$name;passed=$true;clicks=[HarnessWindow]::Down;elapsed=$script:clock})
 }
 if($sourceRow.name -eq 'bounded_sensor_record.ps1'){
  # Execute only the real acknowledgement branch with a mock writer.
  Reset 'persistent_on';$script:lastRecordObservation=Observe ''
  $CoordinationDirectory='mock';$attempted=$true;$errorText='OFF failed'
  function Write-CoordinationMarker([string]$name,[hashtable]$payload){$script:marker=$payload}
  $branch=$ast.Find({param($node) $node -is [Management.Automation.Language.IfStatementAst] -and $node.Extent.Text.StartsWith('if ($CoordinationDirectory -and $attempted)')},$true)
  Assert ($null -ne $branch) 'Missing shutdown acknowledgement branch'
  Invoke-Expression $branch.Extent.Text
  Assert ($script:marker.record_on -eq $true -and $script:marker.error -eq 'OFF failed') 'Failed ACK lost actual ON/error'
  Assert ($script:marker.last_observation.utc -eq 'mock:0') 'Failed ACK lost observation time'
  $results.Add(@{tool=$sourceRow.name;case='failed_ack_preserves_actual_state';passed=$true})
 }
}
@{status='PASS';results=@($results.ToArray());native_ui_calls=0}|ConvertTo-Json -Depth 8 -Compress
"""


@unittest.skipUnless(POWERSHELL, "PowerShell is needed for the fake Win32 harness")
class SensorRecordUiTransitionTests(unittest.TestCase):
    def test_real_powershell_helpers_and_parser_without_native_ui(self):
        sources = [dict(name=name, source=base64.b64encode((TOOLS / name).read_bytes()).decode())
                   for name in ("bounded_sensor_record.ps1", "stop_sensor_record.ps1")]
        setup = "$sources=ConvertFrom-Json '" + json.dumps(sources, separators=(",", ":")) + "'\n"
        # Pipe the harness to avoid Windows' command-line length limit. Its
        # native-facing helper types are replaced with a class without P/Invoke.
        result = subprocess.run(
            [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", "-"],
            input=setup + HARNESS + "\n", encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=40,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        lines = [line[line.index('{"status"'):] for line in result.stdout.splitlines()
                 if '{"status"' in line]
        self.assertTrue(lines, result.stderr + result.stdout)
        report = json.loads(lines[-1])
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["native_ui_calls"], 0)
        self.assertEqual(len(report["results"]), 21)
        self.assertTrue(all(row["passed"] for row in report["results"]))


if __name__ == "__main__":
    unittest.main()
