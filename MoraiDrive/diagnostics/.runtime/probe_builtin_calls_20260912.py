import sys,json,traceback,builtins,time
coordinates=[74.73550037834971,-73.37561682844354]
start=time.monotonic()
try:
 for i in range(3000000):
  for coordinate in coordinates:
   if getattr(coordinate,"ndim",0)!=0:raise ValueError("not scalar")
   value=float(coordinate)
   if value!=coordinate:raise ValueError("wrong value")
 print(json.dumps(dict(status="pass",iterations=i+1,seconds=time.monotonic()-start)))
except Exception:
 print(json.dumps(dict(status="fail",iterations=i+1,traceback=traceback.format_exc(),getattr_binding=repr(builtins.getattr),float_binding=repr(builtins.float))))
 sys.exit(1)
