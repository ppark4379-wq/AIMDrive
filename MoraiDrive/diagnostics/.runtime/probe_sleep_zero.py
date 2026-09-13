import time
end=time.monotonic()+10
i=0
while time.monotonic()<end:
 time.sleep(0)
 i+=1
print(i)
