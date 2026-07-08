import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import signal

seizure_start = 2996 
seizure_end = 3036
fs = 256

with open('chb01-summary.txt', 'r') as f:
    content = f.read()
    print(content)

print("Loading pediatic EEG recordings...")
raw = mne.io.read_raw_edf('chb01_03.edf', preload=True, verbose=False)
print(f"Recording loaded: {raw.info['nchan']}channels at {raw.info['sfreq']} Hz")
print(f"Recording duration: {raw.times[-1]/60:.1f} minutes")
print(f"Seizure occurs at: {seizure_start}s to {seizure_end}s")

pre_start  = (seizure_start - 30) * fs
pre_end    = seizure_start * fs
seiz_start = seizure_start * fs
seiz_end   = seizure_end * fs

data_pre   = raw.get_data(picks='FP1-F7')[0, pre_start:pre_end]
data_seiz  = raw.get_data(picks='FP1-F7')[0, seiz_start:seiz_end]

print(f"Pre-seizure chunk: {len(data_pre)} samples ({len(data_pre)/fs:.0f} seconds)")
print(f"Seizure chunk: {len(data_seiz)} samples ({len(data_seiz)/fs:.0f} seconds)")

t_pre = np.linspace(0, 30, len(data_pre))
t_seiz = np.linspace(0, 40, len(data_seiz))

fig, axes = plt.subplots(2, 1, figsize=(12,6))

axes[0].plot(t_pre, data_pre * 1e6, color='blue')
axes[0].set_title('Pre-Seizure brain activity (30 seconds before seizure) - Child aged 11')
axes[0].set_xlabel('Time (seconds)')
axes[0].set_ylabel('Amplitude(Uv)')

axes[1].plot(t_seiz, data_seiz * 1e6, color='red')
axes[1].set_title('Seizure activity (actual seizure) - Child aged 11')
axes[1].set_xlabel('Time(seconds)')
axes[1].set_ylabel('Amplitude(Uv)')

plt.tight_layout()
plt.savefig('paediatric_seizure_comparision.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved")
def bandpass (data, low, high, fs):
    nyq = fs / 2.0
    b, a = signal.butter(4, [low/nyq, high/nyq], btype='bandpass')
    return signal.filtfilt(b, a, data)

pre_delta = bandpass(data_pre, 0.5, 4, fs)
seiz_delta = bandpass(data_seiz, 0.5, 4, fs)
pre_beta = bandpass(data_pre, 13, 30, fs)
seiz_beta = bandpass(data_seiz, 13, 30, fs)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0,0].plot(t_pre, pre_delta * 1e6, color='blue')
axes[0,0].set_title('Pre-seizure Delta (0.5-4 Hz)')
axes[0,0].set_ylabel('Amplitude (Uv)')

axes[0,1].plot(t_seiz, seiz_delta * 1e6, color='red')
axes[0,1].set_title('Seizure_delta (0.5-4 Hz)')

axes[1,0].plot(t_pre, pre_beta * 1e6, color='blue')
axes[1,0].set_title('Pre-seizure Beta (13-30 Hz)')
axes[1,0].set_ylabel('Amplitude(Uv)')
axes[1,0].set_xlabel('Time (seconds)')

axes[1,1].plot(t_seiz, seiz_beta * 1e6, color='red')
axes[1,1].set_title('Seizure Beta (13-30 Hz)')
axes[1,1].set_xlabel('Time (seconds)')

plt.suptitle('Paediatric EEG - Brain Wave changes during seizure', fontsize=14)
plt.tight_layout()
plt.savefig('paediatric_brain_wave.png', dpi=150, bbox_inches='tight')
plt.show()
print("project6 complete")