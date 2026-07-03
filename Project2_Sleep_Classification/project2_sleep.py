import numpy as np
import matplotlib.pyplot as plt
import mne
raw = mne.io.read_raw_edf('SC4001E0-PSG.edf', preload=True)
print(raw.info)
annotations = mne.read_annotations('SC4001EC-Hypnogram.edf')
raw.set_annotations(annotations)
print("Sleep stages loaded")
print(annotations)
for ann in annotations:
    print(f"{ann['description']} startsat {ann['onset']:.0f} seconds")
    
fs_real= int (raw.info['sfreq'])
    
light_start = 30630 * fs_real
light_end = light_start + (30 * fs_real)
deep_start = 31350 * fs_real
deep_end = deep_start + (30 * fs_real)
    
data_light = raw.get_data(picks='EEG Fpz-Cz')[0, light_start:light_end]
data_deep = raw.get_data(picks='EEG Fpz-Cz')[0, deep_start:deep_end]
    
print(f"Light sleep chunk: {len(data_light)} samples")
print(f"Deep sleep chunk: {len(data_deep)} samples")

t_chunk = np.linspace(0, 30, 3000)
fig, axes = plt.subplots(2,1, figsize=(12, 6))

axes[0].plot(t_chunk, data_light * 1e6, color='blue')
axes[0].set_title('Light Sleep - Stage 1 (30 seconds)')
axes[0].set_xlabel('Time (seconds)')
axes[0].set_ylabel('Amplitude (µV)')

axes[1].plot(t_chunk, data_deep * 1e6, color='green')
axes[1].set_title('Deep Sleep - Stage 4 (30 seconds)')
axes[1].set_xlabel('Time (seconds)')
axes[1].set_ylabel('Amplitude (µV)')

plt.tight_layout()
plt.savefig('light_vs_deep_sleep.png', dpi=150, bbox_inches='tight')
plt.show()
print("Comparision graph saved")

from scipy import signal as sp
def bandpass(data, low_hz, high_hz, fs):
    nyquist = fs / 2.0
    low = low_hz / nyquist
    high = high_hz / nyquist
    b, a = sp.butter(4, [low, high], btype='bandpass')
    return sp.filtfilt(b, a, data)

deep_delta = bandpass(data_deep, 0.5, 4, 100)
light_delta = bandpass(data_light, 0.5, 4, 100)
print('Filters applied to real data')

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

axes[0].plot(t_chunk, light_delta * 1e6, color='blue')
axes[0].set_title('Delta Wave in Light Sleep - Stage 1')
axes[0].set_xlabel('Time (seconds)')
axes[0].set_ylabel('Amplitude (uV)')

axes[1].plot(t_chunk, deep_delta * 1e6, color='darkred')
axes[1].set_title('Delta Wave in Deep Sleep - Stage 4')
axes[1].set_xlabel('Time (seconds)')
axes[1].set_ylabel('Amplitude (uV)')

plt.tight_layout()
plt.savefig('delta_light_vs_deep.png', dpi=150, bbox_inches='tight')
plt.show()
print("Delta comparison saved")

eeg_channel= raw.pick_channels(['EEG Fpz-Cz'])
data, time = raw[0, :3000]
plt.figure(figsize=(2, 4))
plt.plot(time, data[0] * 1e6, color='navy')
plt.title('Real EEG Recording - First 30 seconds (1989)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (µV)')
plt.savefig('real_eeg_sample.png', dpi=150, bbox_inches='tight')
plt.show()