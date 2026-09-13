# Read only the 64-bit Simulator.exe PEB CurrentDirectory field, never environment.
param([Parameter(Mandatory=$true)][string]$TempRoot)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false)
$processes=@(Get-CimInstance Win32_Process -Filter "name='Simulator.exe'")
if ($processes.Count -ne 1) {throw 'Exactly one Simulator.exe required'}
$process=$processes[0]
$code=@'
using System;
using System.Runtime.InteropServices;
public static class SimulatorDirectoryReader {
 [DllImport("kernel32.dll",SetLastError=true)] static extern IntPtr OpenProcess(uint a,bool i,uint p);
 [DllImport("kernel32.dll",SetLastError=true)] static extern bool ReadProcessMemory(IntPtr h,IntPtr p,byte[] b,IntPtr n,out IntPtr r);
 [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr h);
 [DllImport("kernel32.dll",SetLastError=true)] static extern bool IsWow64Process(IntPtr h,out bool w);
 [DllImport("ntdll.dll")] static extern int NtQueryInformationProcess(IntPtr h,int c,IntPtr[] b,int l,out int r);
 static byte[] Read(IntPtr h,long p,int n) {var b=new byte[n];IntPtr r;if(!ReadProcessMemory(h,new IntPtr(p),b,new IntPtr(n),out r)||r.ToInt64()!=n)throw new Exception("ReadProcessMemory failed");return b;}
 public static string Get(uint pid) {
  if(IntPtr.Size!=8)throw new Exception("64-bit reader required");
  IntPtr h=OpenProcess(0x410,false,pid);if(h==IntPtr.Zero)throw new Exception("OpenProcess failed");
  try {bool wow;if(!IsWow64Process(h,out wow)||wow)throw new Exception("64-bit target required");
   var b=new IntPtr[6];int r;if(NtQueryInformationProcess(h,0,b,48,out r)!=0)throw new Exception("PEB query failed");
   long pp=BitConverter.ToInt64(Read(h,b[1].ToInt64()+0x20,8),0);
   byte[] cd=Read(h,pp+0x38,16);int len=BitConverter.ToUInt16(cd,0);long address=BitConverter.ToInt64(cd,8);
   if(len<2||len>32766||len%2!=0)throw new Exception("Invalid CurrentDirectory length");
   return System.Text.Encoding.Unicode.GetString(Read(h,address,len));
  } finally {CloseHandle(h);}
 }
}
'@
# Keep compiler intermediates inside the configured acquisition storage.
$temp=$TempRoot
if (-not (Test-Path -LiteralPath $temp)) {New-Item -ItemType Directory -Path $temp | Out-Null}
$env:TEMP=$temp;$env:TMP=$temp
Add-Type -TypeDefinition $code
@{pid=$process.ProcessId;executable=$process.ExecutablePath;working_directory=[SimulatorDirectoryReader]::Get($process.ProcessId)} | ConvertTo-Json -Compress
