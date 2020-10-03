import winsound as ws
import numpy as np

def Fibonacci(k):
    fib = {}
    for n in range(1,k):
        if n<=0:
            print("Incorrect input")
        # First Fibonacci number is 0
        elif n==1:
            fib[1] = 0
        # Second Fibonacci number is 1
        elif n==2:
            fib[2] = 1
        else:
            fib[n] = fib[n-1]+ fib[n-2]
    return fib

n = 100
fibs = Fibonacci(n)
duration = 300

#### C-Major
for i in range(1,n):
    a = np.mod(fibs[i],8)
    if a == 0:
        ws.Beep(262, duration)
    if a == 1:
        ws.Beep(294, duration)
    if a == 2:
        ws.Beep(330, duration)
    if a == 3:
        ws.Beep(349, duration)
    if a == 4:
        ws.Beep(392, duration)
    if a == 5:
        ws.Beep(440, duration)
    if a == 6:
        ws.Beep(494, duration)
    if a == 7:
        ws.Beep(523, duration)

## A-Minor
##for i in range(1,n):
##    a = np.mod(fibs[i],8)
##    if a == 0:
##        ws.Beep(220, duration)
##    if a == 1:
##        ws.Beep(247, duration)
##    if a == 2:
##        ws.Beep(262, duration)
##    if a == 3:
##        ws.Beep(294, duration)
##    if a == 4:
##        ws.Beep(330, duration)
##    if a == 5:
##        ws.Beep(349, duration)
##    if a == 6:
##        ws.Beep(392, duration)
##    if a == 7:
##        ws.Beep(440, duration)
