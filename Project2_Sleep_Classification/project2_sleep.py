import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import signal
from mne import Epochs, events_from_annotations

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
from mne import Epochs, events_from_annotations
print("Creating 30 second ephocs...")
events, event_id = events_from_annotations(raw)
print(f"Found events: {event_id}")

epochs = Epochs(raw, events, event_id=event_id,
                tmin=0, tmax=30,
                picks='EEG Fpz-Cz',
                baseline=None,
                preload=True)
print(f"Ephocs created: {len(epochs)}")
print(epochs)

X = epochs.get_data()
y = epochs.events[:, 2]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Unique labels: {np.unique(y)}")

X = X.reshape(153, 3001)
print(f"X.reshaped: {X.shape}")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Accuracy: {accuracy* 100:.1f}%")

def bandpower(data, fs, low, high):
    f, psd = signal.welch(data, fs, nperseg=256)
    mask = (f >= low) & (f <= high)
    return np.trapezoid (psd[mask], f[mask])

def hjorth(data):
    activity= np.var(data)
    mobility = np.sqrt(np.var(np.diff(data)) / np.var(data))
    complexity = np.sqrt(np.var(np.diff(np.diff(data))) / np.var(np.diff(data))) / mobility
    return activity, mobility, complexity

def extract_features(epoch, fs):
    delta = bandpower(epoch, fs, 0.5, 4)
    theta = bandpower(epoch, fs, 4, 8)
    alpha = bandpower(epoch, fs, 8, 13)
    beta = bandpower(epoch, fs, 13, 30)
    act, mob, com = hjorth(epoch)
    return [delta, theta, alpha, beta, act, mob, com]

print("Extracting features from all epochs...")
X_features = np.array([extract_features(X[i], 100) for i in range(len(X))])
print(f"Feature matrix shape: {X_features.shape}")

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_features, y, test_size=0.2, random_state=42)
clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf2.fit(X_train2, y_train2)

y_pred2 = clf2.predict(X_test2)
accuracy2 = accuracy_score(y_test2, y_pred2)

print(f"Raw signal accuracy: 41.9%")
print(f"Feature-based accuracy: {accuracy2* 100:.1f}%")
