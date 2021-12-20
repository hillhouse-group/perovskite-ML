import time
import sys

def countdown(t):
    while t>0:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(int(mins), int(secs))
        sys.stdout.write("\r")
        sys.stdout.write(timer) 
        sys.stdout.flush()
        time.sleep(1)
        t -= 1