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
def bandpass(data, low_hz, high_hz):
 nyquist = fs / 2.0
 low = low_hz / nyquist
 high = high_hz / nyquist
 b, a = signal.butter(4, [low, high], btype='bandpass')
 return signal.filtfilt(b, a, data)
delta_f = bandpass(eeg_raw, 0.5, 4)
theta_f = bandpass(eeg_raw, 4, 8)
alpha_f = bandpass(eeg_raw, 8, 13)
beta_f = bandpass(eeg_raw, 13, 30)
print("Filters applied:delta, theta, alpha, beta bands extracted")
fig, axes = plt.subplots(6, 1, figsize=(12, 16))
axes[0].plot(t, eeg_raw, color='navy')
axes[0].set_title('Raw EEG Signal')
axes[0].set_xlabel('Time (seconds)')
axes[0].set_ylabel('Amplitude (uV)')
axes[1].plot(freqs, amps, color='green')
axes[1].set_title('FFT Spectrum')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Amplitude uV')
axes[1].set_xlim(0, 40)
axes[2].plot(t, delta_f, color='blue')
axes[2].set_title('Delta wave Extracted (0.5-4 Hz) - Deep Sleep')
axes[2].set_xlabel('Time (seconds)')
axes[2].set_ylabel('Amplitude (uV)')
axes[3].plot(t, theta_f, color='pink')
axes[3].set_title('Theta wave Extracted (4-8 Hz) - Light Sleep')
axes[3].set_xlabel('Time (seconds)')
axes[3].set_ylabel('Amplitude (uV)')
axes[4].plot(t, alpha_f, color='orange')
axes[4].set_title('Alpha wave Extracted (8-13 Hz) - Relaxed')
axes[4].set_xlabel('Time (seconds)')
axes[4].set_ylabel('amplitude (uV)')
axes[5].plot(t, beta_f, color='red')
axes[5].set_title('Beta wave Extracted (13-30 Hz) - Active Thinking')
axes[5].set_xlabel('Time (seconds)')
axes[5].set_ylabel('Amplitude (uV)')
plt.tight_layout()
plt.show()