$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot '../internal_sensor_record/tools/publish_sensor_record_nas.ps1')
function Assert($ok,$message){if(!$ok){throw $message}}
function Update-Transfer {}
$script:states=@()
function Write-State($phase,$reason){$script:states+=@{phase=$phase;reason=$reason}}
$originalIo=(Get-Item Function:Invoke-NasIo).ScriptBlock
$script:moves=0
function Invoke-NasIo($operation,$path,$action){
 if($operation -eq 'publish_directory'){$script:moves++}
 & $originalIo $operation $path $action
}
Add-Type @'
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using Microsoft.Win32.SafeHandles;
public sealed class NasDirectoryLockFixture : IDisposable {
 [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)]
 static extern SafeFileHandle CreateFile(string name,uint access,uint share,IntPtr security,uint creation,uint flags,IntPtr template);
 SafeFileHandle handle;
 Timer timer;
 public NasDirectoryLockFixture(string path,int releaseAfterMilliseconds) {
  handle=CreateFile(path,1,3,IntPtr.Zero,3,0x02000000,IntPtr.Zero);
  if(handle.IsInvalid)throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
  if(releaseAfterMilliseconds>=0)timer=new Timer(_=>handle.Dispose(),null,releaseAfterMilliseconds,Timeout.Infinite);
 }
 public void Dispose(){if(timer!=null)timer.Dispose();handle.Dispose();}
}
'@
$fixture=Join-Path $env:TEMP ('morai-publish-retry-'+[Guid]::NewGuid().ToString('N'))
$null=[IO.Directory]::CreateDirectory($fixture)
$held=$null
try {
 $src=Join-Path $fixture 'transient'
 $dst=Join-Path $fixture 'transient-final'
 $null=[IO.Directory]::CreateDirectory($src)
 [IO.File]::WriteAllText((Join-Path $src 'payload.txt'),'preserved')
 $held=[NasDirectoryLockFixture]::new($src,200)
 Publish-NasDirectory $src $dst -retryDelayMilliseconds 100
 $held.Dispose();$held=$null
 Assert ($script:moves -gt 1 -and $script:moves -le 5) 'Real directory lock must retry then recover'
 Assert ([IO.File]::ReadAllText((Join-Path $dst 'payload.txt')) -eq 'preserved') 'Contents preserved'
 Assert ($script:states.Count -gt 0 -and $script:states[0].phase -eq 'publishing') 'Retry stage observable'

 $script:moves=0
 $src=Join-Path $fixture 'persistent';$dst=Join-Path $fixture 'persistent-final'
 $null=[IO.Directory]::CreateDirectory($src)
 $held=[NasDirectoryLockFixture]::new($src,-1)
 $failed=$false
 try{Publish-NasDirectory $src $dst -maxAttempts 3 -retryDelayMilliseconds 10}
 catch{$failed=$true;Assert ($_.Exception.Message -match 'after 3 attempts.*Win32=32') 'Precise exhaustion reason'}
 Assert $failed 'Persistent lock must fail'
 Assert ($script:moves -eq 3) 'Retry count bounded'
 Assert ([IO.Directory]::Exists($src) -and ![IO.Directory]::Exists($dst)) 'Pending preserved without false publication'
 $held.Dispose();$held=$null

 $script:moves=0
 $dst=Join-Path $fixture 'existing-final'
 $null=[IO.Directory]::CreateDirectory($dst)
 [IO.File]::WriteAllText((Join-Path $dst 'keep.txt'),'existing')
 $failed=$false
 try{Publish-NasDirectory $src $dst -retryDelayMilliseconds 0}catch{$failed=$true}
 Assert ($failed -and $script:moves -eq 1) 'Existing destination must fail without retry'
 Assert ([IO.File]::ReadAllText((Join-Path $dst 'keep.txt')) -eq 'existing') 'Existing destination untouched'

 $script:moves=0
 function Invoke-NasIo($operation,$path,$action){
  if($operation -eq 'publish_directory'){$script:moves++;throw [UnauthorizedAccessException]::new('Injected denied access')}
  & $originalIo $operation $path $action
 }
 $failed=$false
 try{Publish-NasDirectory $src (Join-Path $fixture 'denied') -retryDelayMilliseconds 0}catch{$failed=$true}
 Assert ($failed -and $script:moves -eq 1) 'Permission failures never retried'
 Write-Host 'PASS: real transient/persistent directory locks, bounded retries, no overwrite, permission errors'
} finally {
 if($null -ne $held){$held.Dispose()}
 $resolved=[IO.Path]::GetFullPath($fixture)
 $temp=[IO.Path]::GetFullPath($env:TEMP).TrimEnd('\')+'\'
 if(!$resolved.StartsWith($temp,[StringComparison]::OrdinalIgnoreCase) -or [IO.Path]::GetFileName($resolved) -notlike 'morai-publish-retry-*'){throw 'Unsafe fixture cleanup'}
 Remove-Item -LiteralPath $resolved -Recurse -Force
}
