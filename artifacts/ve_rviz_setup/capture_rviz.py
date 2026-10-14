import ctypes
import re
import subprocess
import time
from PyQt5.QtWidgets import QApplication

tree = subprocess.check_output(['xwininfo','-root','-tree']).decode()
match = re.search(r'(0x[0-9a-f]+) "ve_roi_base_link[^"\n]*"',tree)
if not match:
    raise SystemExit('VE ROI RViz window not found')
window = int(match.group(1),16)
lib = ctypes.cdll.LoadLibrary('libX11.so.6')
lib.XOpenDisplay.restype = ctypes.c_void_p
display = lib.XOpenDisplay(None)
lib.XRaiseWindow.argtypes = [ctypes.c_void_p,ctypes.c_ulong]
lib.XSetInputFocus.argtypes = [ctypes.c_void_p,ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
lib.XFlush.argtypes = [ctypes.c_void_p]
lib.XRaiseWindow(display,window)
lib.XSetInputFocus(display,window,2,0)
lib.XFlush(display)
time.sleep(.3)
app = QApplication([])
image = app.primaryScreen().grabWindow(window)
target = '/mnt/c/Users/dydwn/AIMDrive/artifacts/ve_rviz_setup/rviz_window.png'
print('saved',image.save(target),image.width(),image.height())
