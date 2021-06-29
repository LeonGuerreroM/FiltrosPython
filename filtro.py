import numpy as np
from scipy import signal

h = [1,2,3,4]

N = len(h)
input_sample = np.linspace(1,10,10,endpoint=True)
Nx = len(input_sample)

res = signal.convolve(input_sample,h)
x = np.zeros(Nx+N+1)

for n in range(0,Nx+N-1,1):
    if(n<Nx):
        x[0]=input_sample[n]
    else:
        x[0] = 0
    y=0.0
    for k in range(0,N,1):
        y = y+h[k]*x[k]
    
    for m in range(N-1,0,-1):
        x[m]=x[m-1]
    
    print(y)
    print(res)