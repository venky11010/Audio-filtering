import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import signal

def custom_filtfilt(b, a, x):
    # Initialize output array with zeros
    y = np.zeros_like(x)
    
    # Apply forward filter
    for n in range(len(x)):
        y[n] = b[0]*x[n] + b[1]*x[n-1] + b[2]*x[n-2] + b[3]*x[n-3]
        if n > 0:
            y[n] -= a[1]*y[n-1]
        if n > 1:
            y[n] -= a[2]*y[n-2]
        if n > 2:
            y[n] -= a[3]*y[n-3]
    
    # Apply backward filter
    y_backward = np.zeros_like(x)
    for n in range(len(x)-1, -1, -1):
        y_backward[n] = b[0]*y[n] + b[1]*y[n-1] + b[2]*y[n-2] + b[3]*y[n-3]
        if n < len(x) - 1:
            y_backward[n] -= a[1]*y_backward[n+1]
        if n < len(x) - 2:
            y_backward[n] -= a[2]*y_backward[n+2]
        if n < len(x) - 3:
            y_backward[n] -= a[3]*y_backward[n+3]
    
    # Combine forward and backward filters to get final result
    y_final = (y + y_backward) / 2
    
    return y_final

# Coefficients of the filter
a = [1, -1.87302725, 1.30032695, -0.31450204]
b = [0.01409971, 0.04229913, 0.04229913, 0.01409971]

# Load .wav file
input_wav_file = 'music7_cut.wav'
x, sr = librosa.load(input_wav_file, sr=None)  # sr=None to preserve the original sampling rate

n = np.arange(1, len(x), 10000)

X = np.fft.fft(x)

y_custom = custom_filtfilt(b, a, x)

Y = np.fft.fft(y_custom)

H = [y/x for x, y in zip(X, Y)]

H_magnitude = [np.abs(h) for h in H]

angular_freq = np.linspace(0, 4 * np.pi, len(H_magnitude))
freq_bins = np.fft.fftfreq(len(H_magnitude), d=1/sr)

plt.plot(freq_bins, H_magnitude)
plt.xlabel('Index')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()
