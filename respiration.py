import pyaudio
import wave
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import scipy

# length of data to read.
chunk = 65536



'''
************************************************************************
      This is the start of the "minimum needed to read a wave"
************************************************************************
'''
# open the file for reading.
def get_frames(file):
    wf = wave.open(file, 'rb')
    channel=wf.getnchannels()
    print(channel)
    # create an audio object
    p = pyaudio.PyAudio()
    # cleanup stuff.   
    p.terminate()
    # read data (based on the chunk size)
    data = wf.readframes(-1)
    return data

def convert_bytes_to_inputs():
    data=get_frames("09102020-11-46-50-0.wav")
    data = np.frombuffer(data, dtype=np.int16) 
    return data

def two_waves_delete_mean_signal():
    #divide signal by four data of microphones + Zero padding+delete mean signal
    data=convert_bytes_to_inputs()
    data=list(data)
    mean=np.mean(data)
    j=0
    print(len(data))
    while(j<len(data)):
        if j%1000000==0 :
            print(j)
        data[j]=data[j]-mean
        j=j+1
    print("ok1")
    NFFT = 65535
    sample_rate = 192000
    deltaT = 1/sample_rate
    deltaF = sample_rate/NFFT
    print("ok2")
    data1, data2, data3, data4 = data[0::4], data[1::4], data[2::4], data[3::4] #tous les deux éléments pairs data1, tous les deux éléments impairs data2
    #data1, data2 = data[0::2], data[1::2]#tous les deux éléments pairs data1, tous les deux éléments impairs data2
    datas1=[]
    datas2=[]
    datas3=[]
    datas4=[]
    i=1
    print("ok3")
    while(i*chunk<len(data1)):
        datas1.append(data1[(i-1)*chunk:i*chunk])
        datas2.append(data2[(i-1)*chunk:i*chunk])
        datas3.append(data3[(i-1)*chunk:i*chunk])
        datas4.append(data4[(i-1)*chunk:i*chunk])
        i=i+1

    return datas1, datas2, datas3, datas4
    #return datas1, datas2


def kaiser():
    data1, data2, data3, data4=two_waves_delete_mean_signal()
    #data1, data2=two_waves_delete_mean_signal()
    kaiser=np.kaiser(chunk, 14)
    data1=data1*kaiser
    data2=data2*kaiser
    data3=data3*kaiser
    data4=data4*kaiser
    return data1, data2, data3, data4
    #return data1, data2

#cacul fft magnitude
def fft_magnitude():
    data1, data2, data3, data4=kaiser()
    #data1, data2=kaiser()
    i=0
    fftdata1=[]
    fftdata2=[]
    fftdata3=[]
    fftdata4=[]
    while(i<len(data1)):
        fftdata1.append(np.abs(scipy.fftpack.fft(data1[i])))
        fftdata2.append(np.abs(scipy.fftpack.fft(data2[i])))
        fftdata3.append(np.abs(scipy.fftpack.fft(data3[i])))
        fftdata4.append(np.abs(scipy.fftpack.fft(data4[i])))
        i=i+1
    return fftdata1, fftdata2, fftdata3, fftdata4
    #return fftdata1, fftdata2

def plot_fft(fftdata, color):
    NFFT = 65535
    sample_rate = 192000
    listeframes=np.linspace(0.0, sample_rate/2, NFFT/2)
    i=0
    while(i<len(fftdata)):
        plt.plot(listeframes[12969:14336], fftdata[i][12969:14336], c=color)
        plt.show()
        i=i+1

def plot_fft_time(fftdata, color):
    NFFT = 65535
    sample_rate = 192000
    deltaT = 1/sample_rate
    deltaF = sample_rate/NFFT
    i=0
    listeframes=np.linspace(0, sample_rate//2, NFFT//2)
    listeframes=listeframes[12969:14336]
    print("plot")
    pointx=[]
    pointy=[]
    rates=[]
    time=0
    r=0
    rbool=False
    while(i<len(fftdata)):
        fftdata[i]=fftdata[i][12969:14336]
        j=0
        while j<len(fftdata[i]):
            if fftdata[i][j]>800  :
                pointx.append(listeframes[j])
                pointy.append(i*300)
                rbool=True
            j=j+1
        if rbool==True :
            r=r+1
        time=time+300
        if time>=1000 :
            rates.append(r)
            r=0
            time=time-1000
            if rbool==True :
                r=r+1
        rbool=False
        i=i+1

    file1=open("fft_respiration_zootopie.txt", "w")
    i=0
    while(i<len(pointx)) :
        txt=str(pointx[i])+" "+str(pointy[i])+"\n"
        file1.write(txt)
        i=i+1
    file1.close()
    print(time)
    print(rates)
    plt.plot(pointx, pointy, c=color)
    plt.xlabel("Fréquences (Hertz)")
    plt.ylabel("Temps (ms)")
    plt.show()

def plots_fft():
    #fftdata1, fftdata2, fftdata3, fftdata4=fft_magnitude()
    fftdata1, fftdata2=fft_magnitude()
    print("plot")
    plot_fft(fftdata1, 'blue')
    plot_fft(fftdata2, 'red')
    #plot_fft(fftdata3, 'green')
    #plot_fft(fftdata4, 'purple')

def plots_fft_time():
    fftdata1, fftdata2, fftdata3, fftdata4=fft_magnitude()
    #fftdata1, fftdata2=fft_magnitude()
    plot_fft_time(fftdata1, 'blue')
    plot_fft_time(fftdata2, 'red')
    plot_fft_time(fftdata3, 'green')
    plot_fft_time(fftdata4, 'purple')

 
plots_fft_time()
