# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 12:15:15 2021

@author: equipo
"""

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves
from scipy.fftpack import fft, fftfreq
import winsound
import time

#*******************************Archivo Audio**********************************************
def plot_response(fs,w,h,title):
    "Utility function to plot response functions"
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)))
    ax.set_ylim(-40, 5)
    ax.set_xlim(0, 0.5*fs)
    ax.grid(True)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(title)
    plt.show()
    
    
    ax = fig.add_subplot(212)
    angles = np.unwrap(np.angle(h))
    plt.plot((fs*0.5/np.pi)*w, angles)
    #plt.xscale('log')
    plt.title('Phase response')
    plt.xlabel('Frecuency [Hz]')
    plt.ylabel('radianes')
    plt.margins(0,0.1)
    plt.grid(which='both', axis='both')
    time.sleep(0.5)
    plt.show()
   
plt.close('all')

#archivo = input('archivo de sonido:' )
#archivo = 'muestra01_ElAguacateIntro.wav'
archivo = 'bobinasDeTeslaHimnoCCCP_4TM1.wav'
FS, sonido = waves.read(archivo)

fs = FS     # Sample rate, Hz


'''
band=[6500, 7500]
trans_width= 100
numtaps=375 #Size

edges = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5*fs]
taps = signal.remez(numtaps, edges, [1, 0, 1], Hz=fs)
w, h = signal.freqz(taps, [1], worN=2000)


'''
# canales: monofónico o estéreo
tamano = np.shape(sonido)
muestras = tamano[0]
m = len(tamano)
canales = 1  # monofónico
if (m>1):  # estéreo
    canales = tamano[1]
# experimento con un canal
if (canales>1):
    canal = 0
    uncanal = sonido[:,canal] 
else:
    uncanal = sonido
    
# rango de observación en segundos
inicia = 5.00
termina = 10.00
# observación en número de muestra
a = int(inicia*FS)
b = int(termina*FS)
parte = uncanal[a:b]

#res = signal.convolve(taps, parte)


# tiempos en eje x
dt = 1/FS
ta = a*dt
tb = (b)*dt
tab = np.arange(ta,tb,dt)

'''plt.figure(1)
plt.clf()
plt.plot(tab,parte,'b')
plt.xlabel('tiempo(s)')
plt.ylabel('Amplitud')
plt.title('Señal sin filtar en el dominio del tiempo')'''



fig = plt.figure(figsize=(12, 6))
Y = fft(parte)/len(parte)  # Transformada normalizada
frq2 = fftfreq(len(Y), dt)
ax4 = fig.add_subplot(111)
ax4.vlines(frq2, 0, abs(Y),'b') # Espectro de amplitud
plt.xlim(0, FS*0.5)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud')
plt.title('Señal no filtrada en la frecuencia')
plt.grid(which='both', axis='both')
plt.show()
#time.sleep(0.5)
#****************************Stop Band FIR Filter*************************************
fs = FS     # Sample rate, Hz


band=[1900, 2150]
trans_width= 10
numtaps=375 #Size

edges = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5*fs]
taps = signal.remez(numtaps, edges, [1, 0, 1], Hz=fs)
w, h = signal.freqz(taps, [1], worN=2000)
#44100
res = signal.convolve(taps, parte)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)),label="Filter",linewidth=2)
ax.set_ylim(-80, 5)
ax.set_xlim(0, 7500)
ax.grid(True)
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Ganancia [dB]')
ax.set_title('Respuesta en frecuencia del Filtro 1')
plt.show()
    
'''plt.figure(4)
plt.clf()
#ax = fig.add_subplot(212)
angles = np.unwrap(np.angle(h))
plt.plot((fs*0.5/np.pi)*w, angles,'b')
#plt.xscale('log')
plt.title('Phase response')
plt.xlabel('Frecuency [Hz]')
plt.ylabel('radianes')
plt.margins(0,0.1)
plt.grid(which='both', axis='both')
time.sleep(0.5)
plt.show()

plt.figure(5)
plt.clf()
t = np.linspace(0, (len(taps) - 1), len(taps)) # Intervalo de tiempo en segundos
plt.step(t,taps,'b')
plt.grid()
plt.title(' Filter Impulse response ')
plt.xlabel('n [samples]')
plt.ylabel('Amplitude')
plt.show()   '''


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4
'''
'''
band=[3450, 3700]
trans_width= 100
numtaps=375 #Size

edges = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5*fs]
taps = signal.remez(numtaps, edges, [1, 0, 1], Hz=fs)
w, h = signal.freqz(taps, [1], worN=2000)

res = signal.convolve(taps, res, mode = 'full')
'''
'''
#$$$$$$$$$$$$$$$$$$$$$
res2 = res.astype(np.int16)
t = np.linspace(0, (len(res)-1), len(res))
#res = signal.convolve(taps, res, mode = 'full')




archivo1 = 'teslita1.wav'
waves.write(archivo1,FS,res)
'''
archivo1 = 'archivoDSP14_CI_001_FiltradaFIR.wav'
waves.write(archivo1,FS,res)
'''
#****************************Stop Band FIR Filter Plot*************************************

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)),label="Filter",linewidth=2)
ax.set_ylim(-80, 5)
ax.set_xlim(0, 8500)
ax.grid(True)
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Ganancia [dB]')
ax.set_title('Respuesta en frecuencia del Filtro 2')
plt.show()
    
'''plt.figure(8)
plt.clf()
#ax = fig.add_subplot(212)
angles = np.unwrap(np.angle(h))
plt.plot((fs*0.5/np.pi)*w, angles,'b')
#plt.xscale('log')
plt.title('Phase response')
plt.xlabel('Frecuency [Hz]')
plt.ylabel('radianes')
plt.margins(0,0.1)
plt.grid(which='both', axis='both')
time.sleep(0.5)
plt.show()
plt.figure(9)
plt.clf()
t = np.linspace(0, (len(taps) - 1), len(taps)) # Intervalo de tiempo en segundos
plt.step(t,taps,'b')
plt.grid()
plt.title(' Filter Impulse response ')
plt.xlabel('n [samples]')
plt.ylabel('Amplitude')
plt.show()   '''



archivo1 = 'archivoDSP14_CI_001_FiltradaFIR.wav'
waves.write(archivo1,FS,res)


'''plt.figure(10)
plt.clf()
plt.plot(res2,'b')
plt.xlabel('tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Señal filtrada en el dominio del tiempo')
plt.grid(which='both', axis='both')
plt.show()'''


fig = plt.figure(figsize=(12, 6))
filtrada = fft(res2)/len(res2)  # Transformada normalizada
frq3 = fftfreq(len(filtrada), 1/FS)
ax4 = fig.add_subplot(111)
ax4.vlines(frq3, 0, abs(filtrada),'b') # Espectro de amplitud
plt.xlim(0, fs/2)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud')
plt.title('Señal filtrada en la frecuencia')
plt.grid(which='both', axis='both')
plt.show()
time.sleep(0.5)

archivo2 = 'teslita2.wav'
#archivo2 = 'audiofiltradofIR.wav'
waves.write(archivo2,FS,res2)

