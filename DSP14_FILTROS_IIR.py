# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:02:17 2021

@author: León Mora
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves
from scipy import signal
from scipy.fftpack import fft, fftfreq
import winsound
import scipy.io.wavfile as waves
import time
import matplotlib 
plt.close('all')
archivo = 'archivoDSP14_CI_001.wav'
muestreo, sonido = waves.read(archivo)


fn = muestreo/2
fp1 = 1850
fr1 = 1900
fr2 = 2100
fp2 = 2150

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
    

inicia = 0.00
termina = 15.00


aa = int(inicia*muestreo)
bb = int(termina*muestreo)
parte = uncanal[aa:bb]
dt = 1/muestreo
ta = aa*dt
tb = (bb)*dt
tab1 = np.arange(ta,tb,dt)


plt.figure(1)
plt.clf()
plt.plot(tab1,parte)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Señal sin filtrar en el tiempo')
plt.show()



fig = plt.figure(figsize=(12, 6))
Y = fft(parte)/len(parte)  
frq1 = fftfreq(len(Y), dt)
ax4 = fig.add_subplot(111)
ax4.vlines(frq1, 0, abs(Y)) 
plt.xlim(0, muestreo/2)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.title('Señal no filtrada en la frecuencia')
plt.show()

#Filtro 1 
plt.figure()
plt.clf()
b, a = signal.cheby2(4, 50,[2*fr1/muestreo,2*fr2/muestreo], 'bandstop', analog=False)
w, h = signal.freqz(b, a)
plt.plot((muestreo*0.5/np.pi)*w, abs(h))
plt.title('Respuesta en frecuencia del Chebyshev I')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud normalizada [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.show()

plt.figure()
plt.clf()
angles = np.unwrap(np.angle(h))
plt.plot((muestreo*0.5/np.pi)*w, angles)
plt.title('Respuesta en fase del Chebyshev I')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('radianes')
plt.margins(0, 0.1)
plt.show()
time.sleep(0.5)
yy=signal.lfilter(b, a, parte)
yy2=yy.astype(np.int16)

plt.figure()
plt.clf()
imp = signal.dlti(*signal.cheby2(4, 50,[2*fr1/muestreo,2*fr2/muestreo], 'bandstop', analog=False))
n, y = signal.dimpulse(imp, n=250)
plt.stem(n, np.squeeze(y))
plt.grid()
plt.title('Respuesta al impulso del Chebyshev I')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')

yy3=yy2
sonido=yy3

#Filtro 2 
fp1 = 6900
fr1 = 6950
fr2 = 7050
fp2 = 7100


n, wn = signal.cheb2ord([fp1/fn, fp2/fn], [fr1/fn, fr2/fn], 3, 50)
b, a = signal.cheby2(n, 50, wn, 'bandstop')

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
    

inicia = 0.00
termina = 15.00


aa = int(inicia*muestreo)
bb = int(termina*muestreo)
parte = uncanal[aa:bb]


yy=signal.lfilter(b, a, parte)
yy2=yy.astype(np.int16)

dt = 1/muestreo
ta = aa*dt
tb = (bb)*dt
tab1 = np.arange(ta,tb,dt)



plt.figure()
plt.clf()


w, h = signal.freqz(b, a)
plt.plot((muestreo*0.5/np.pi)*w, 20 * np.log10(abs(h)))
plt.title('Respuesta en frecuencia del Chebyshev II')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.show()

plt.figure()
plt.clf()
FS=muestreo
b, a = signal.cheby2(12, 40, 2*2200/FS, 'lowpass', analog=False)
w, h = signal.freqz(b, a)
angles = np.unwrap(np.angle(h))
plt.plot((FS*0.5/np.pi)*w, angles)
plt.title('Respuesta en fase del Chebyshev II')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud normalizada [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(2200, color='green') 
plt.show()

plt.figure()
plt.clf()
imp = signal.dlti(*signal.cheby2(4, 50,[2*fr1/muestreo,2*fr2/muestreo], 'bandstop', analog=False))
n, y = signal.dimpulse(imp, n=250)
plt.stem(n, np.squeeze(y))
plt.grid()
plt.title('Respuesta al impulso del Chebyshev II')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')

plt.figure()
plt.clf()
plt.plot(tab1,yy2)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Señal filtrada en el tiempo')
plt.show()


fig = plt.figure(figsize=(12, 6))
Y2 = fft(yy2)/len(yy2)  
frq2 = fftfreq(len(Y2), dt)
ax4 = fig.add_subplot(111)
ax4.vlines(frq2, 0, abs(Y2)) 
plt.xlim(0, muestreo/2)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud')
plt.title('Señal filtrada en la frecuencia')
plt.show()
time.sleep(0.5)
audfil=yy2
archfil='archivoDSP14_CI_001_FiltradaIIR.wav'
waves.write(archfil, muestreo,audfil)