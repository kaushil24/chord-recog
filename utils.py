import matplotlib.pyplot as plt
import librosa as ls
import librosa.display
import numpy as np


class AudioUtils():
    
    def __init__(self, n_fft = 2048, hop_length = 512, sr = 41000):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
    
    def makeSpectogram(self, sig, n_fft = None, hop_length = None):
        '''
        Takes input as signal and returns it's spectogram
        '''
        if n_fft == None: n_fft = self.n_fft
        if hop_length == None: hop_length = self.hop_length
        # short term fourier transform
        stft = ls.stft(sig, n_fft = n_fft, hop_length = hop_length, win_length = n_fft, window = 'hann')
        spectogram_librosa = np.abs(stft)
        
        # Converting amplitude to dB (log conversion)
        spectogram_librosa_db = ls.power_to_db(spectogram_librosa **2, ref = np.max)
        return spectogram_librosa_db
    
    def fileToSpecto(self, file, sr = None, n_fft = None, hop_length = None, show = False, title = 'Spectogram'):
        '''
        Input: File name and locr.
        Returns spectogram
        '''
        if n_fft is None: n_fft = self.n_fft 
        if hop_length is None: hop_length = self.hop_length
        if sr is None: sr = self.sr
        
        sig, sr = ls.load(file, sr = 41000)
        specto = self.makeSpectogram(sig, n_fft, hop_length)
        if show:
            self.showSpecto(specto, sr = sr, hop_length = hop_length, title = title)
        return specto
    
    def showSpecto(self, specto,sr = None , hop_length=None,title = 'Spectogram', xlabel = 'Time', ylabel = 'Hz', y_axis = 'log'):
        '''
        '''
        if sr is None: sr = self.sr 
        if hop_length is None: hop_length = self.hop_length 
        
        ls.display.specshow(specto, sr = sr, hop_length = hop_length, y_axis='log', x_axis='time')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar(format = '%+2.0f dB')
        plt.tight_layout()
        plt.show()