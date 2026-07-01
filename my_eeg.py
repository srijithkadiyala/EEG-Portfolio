import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

print("EEG script is running")
fs= 256
duration = 10
t = np .linspace(0, duration, fs * duration, endpoint=False)
delta = 1.5 * np.sin(2 * np.pi * 2 * t)
theta = 1.0 * np.sin(2 * np.pi * 6 * t)
alpha = 2.0 * np.sin(2 * np.pi * 10 * t)
beta = 0.5 * np.sin(2 * np.pi * 20 * t)
np.random.seed(42)
noise = 0.4 * np.random.randn(len(t))
eeg_raw = delta + theta + alpha + beta + noise
print(f"Signal created: {len(eeg_raw)} samples, {duration} seconds at {fs} Hz")
N = len(eeg_raw)
yf = fft(eeg_raw)
xf = fftfreq(len(eeg_raw), d=1/fs)
half = N // 2
freqs = xf[:half]
amps = (2.0 / N) * np.abs(yf[:half])
print(f"FFT complete: {len(freqs)} frequency bins")
