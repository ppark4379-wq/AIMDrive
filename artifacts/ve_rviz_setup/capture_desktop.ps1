Add-Type @'
using System;
using System.Text;
using System.Runtime.InteropServices;
public class VeWindowCapture {
 public delegate bool EnumProc(IntPtr w, IntPtr p);
 [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc f, IntPtr p);
 [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr w, StringBuilder b, int n);
 [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr w, int state);
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr w);
 public static void Focus() {
  EnumWindows((w,p)=>{var text=new StringBuilder(1024); GetWindowText(w,text,1024);
    if(text.ToString().Contains("ve_roi_base_link")) {ShowWindow(w,3);SetForegroundWindow(w);return false;}return true;},IntPtr.Zero);
 }
}
'@
[VeWindowCapture]::Focus()
Start-Sleep -Milliseconds 500
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$screenRect = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$screenBitmap = New-Object System.Drawing.Bitmap($screenRect.Width, $screenRect.Height)
$screenGraphics = [System.Drawing.Graphics]::FromImage($screenBitmap)
$screenGraphics.CopyFromScreen($screenRect.Location, [System.Drawing.Point]::Empty, $screenRect.Size)
$screenBitmap.Save('C:\Users\dydwn\AIMDrive\artifacts\ve_rviz_setup\rviz_final_desktop.png')
$screenGraphics.Dispose()
$screenBitmap.Dispose()
