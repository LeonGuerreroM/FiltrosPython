import numpy as np
from scipy import signal

a = [1,2,3,4]
b = [1,2,3,4]

Na = len(a)
input_sample = np.linspace(1,10,10,endpoint=True)
Nx = len(input_sample)
yy = signal.lfilter(b,a,input_sample)

u = np.zeros(Na)

for n in range(0,Nx,1):
    x = input_sample[n]
    temp1 = 0
    temp2 = 0
    for k in range(1,Na,1):
        temp1 = temp1 - a[k]*u[k]
        temp2 = temp2 + b[k]*u[k]
    u[0] = x + temp1
    y = u[0]*b[0]+temp2
    
    for m in range(Na-1,0,-1):
        u[m] = u[m-1]
        
    print(y)
    print(yy)
    print("Oprime enter")