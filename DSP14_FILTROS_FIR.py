# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:04:07 2021

@author: León Mora
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves
from scipy import signal
from scipy.fftpack import fft, fftfreq
import winsound
#import scipy.io.wavfile as waves
import time
def plot_response(fs, w, h, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)))
    ax.set_ylim(-40, 5)
    ax.set_xlim(0, 0.5*fs)
    ax.grid(True)
    ax.set_xlabel('Frecuencia [Hz]')
    ax.set_ylabel('Ganancia [dB]')
    ax.set_title(title)
    
plt.close('all')


archivo = 'BobinasDeTeslaHimnoCCCP_4TM1.wav'
muestreo, sonido = waves.read(archivo)



fn = muestreo/2
fp1 = 925
fr1 = 950
fr2 = 1050
fp2 = 1075

n, wn = signal.cheb2ord([fp1/fn, fp2/fn], [fr1/fn, fr2/fn], 3, 40)
b, a = signal.cheby2(n, 40, wn, 'bandstop')

tamano = np.shape(sonido)
muestras = tamano[0]
m = len(tamano)
canales = 1
if (m>1):
    canales = tamano[1]

if (canales>1):
    canal = 0
    uncanal = sonido[:,canal] 
else:
    uncanal = sonido
    

inicia = 5.00
termina = 10.00


aa = int(inicia*muestreo)
bb = int(termina*muestreo)
parte = uncanal[aa:bb]
dt = 1/muestreo
ta = aa*dt
tb = (bb)*dt
tab1 = np.arange(ta,tb,dt)

'''plt.figure(1)
plt.clf()
plt.plot(tab1,parte)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Señal sin filtrar en el tiempo')
plt.show()'''

fig = plt.figure(figsize=(12, 6))
Y = fft(parte)/len(parte)  
frq1 = fftfreq(len(Y), dt)
ax4 = fig.add_subplot(111)
ax4.vlines(frq1, 0, abs(Y))
plt.xlim(0, muestreo/2)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud')
plt.title('Señal no filtrada en la frecuencia')
plt.show()

time.sleep(0.5)


#Filtro 1 
fn = muestreo/2
fs = fn  
band = [fr1, fr2]  
trans_width = 200    
numtaps = 175
edges = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5*fs]
taps = signal.remez(numtaps, edges, [1, 0, 1], Hz=fs)
w, h = signal.freqz(taps, [1], worN=2000)
plot_response(fs, w, h, "Respuesta en frecuencia del FIR I")
    
'''plt.show()
plt.figure()
angles = np.unwrap(np.angle(h))
plt.plot((fs*0.5/np.pi)*w, angles)
plt.title('Respuesta en fase del FIR I')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('radianes')
plt.margins(0, 0.1)
plt.show()


plt.figure()
plt.clf()
t = np.linspace(0, (len(taps) - 1), len(taps)) 
plt.step(t,taps)
plt.grid()
plt.title('Respuesta al impulso del FIR I')
plt.xlabel('Muestras')
plt.ylabel('Amplitude') '''

res= signal.convolve(taps, parte, mode = 'full')
res2= res.astype(np.int16)
res3=res2
t = np.linspace(0, (len(res)-1), len(res))


#Filtro 2 
fp1 = 3350
fr1 = 3400
fr2 = 3700
fp2 = 3750

fn = muestreo/2
fs = fn      
band = [fr1, fr2]

trans_width = 100    
numtaps = 75
edges = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5*fs]
print(fs)
print (edges)
taps = signal.remez(numtaps, edges, [1, 0, 1], Hz=fs)

w, h = signal.freqz(taps, [1], worN=2000)
plot_response(fs, w, h, "Respuesta en frecuencia del FIR II")
plt.show()

'''plt.figure()
angles = np.unwrap(np.angle(h))
plt.plot((fs*0.5/np.pi)*w, angles)
plt.title('Respuesta en fase del FIR II')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('radianes')
plt.margins(0, 0.1)
plt.show()


plt.figure()
plt.clf()
t = np.linspace(0, (len(taps) - 1), len(taps)) 
plt.step(t,taps)
plt.grid()
plt.title('Respuesta al impulso del FIR II')
plt.xlabel('Muestras')
plt.ylabel('Amplitude')''' 


res = signal.convolve(taps, res3, mode = 'full')
res2 = res.astype(np.int16)
t = np.linspace(0, (len(res)-1), len(res))



'''plt.figure()
plt.clf()
plt.plot(t,res2)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Señal filtrada en el tiempo')
plt.show()'''

fig = plt.figure(figsize=(12, 6))
filtrada = fft(res2)/len(res2)  
frq3 = fftfreq(len(filtrada), 1/fs)
ax4 = fig.add_subplot(111)
ax4.vlines(frq3, 0, abs(filtrada)) 
plt.xlim(0, fs/2)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud')
plt.title('Señal filtrada en la frecuencia')
plt.show()
audfil=res2
archfil='MG_Tesla.wav'
waves.write(archfil, muestreo,audfil)