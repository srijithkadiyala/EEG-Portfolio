import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

data_path = 'Bonn University Dataset'
normal_file = os.path.join(data_path, "Z", "Z001.txt")
seizure_file = os.path.join(data_path, "S", "S001.txt")

normal = np.loadtxt(normal_file)
seizure = np.loadtxt(seizure_file)

print(f"Normal recording: {len(normal)} samples")
print(f"Seizure recording: {len(seizure)} samples")

fs_bonn = 173.6
t = np.linspace(0, 23.6, 4097)

fig, axes = plt.subplots(2,1, figsize=(12, 6))
axes[0].plot(t, normal , color='blue')
axes[0].set_title('Normal brain activity (Normal person)')
axes[0].set_xlabel('Time [seconds]')
axes[0].set_ylabel('Amplitude (uV)')

axes[1].plot(t, seizure, color='red')
axes[1].set_title('Seizure brain activity (Epileptic patient)')
axes[1].set_xlabel('Time [seconds]')
axes[1].set_ylabel('Amplitude (uV)')

plt.tight_layout()
plt.savefig('normal_vs_seizure.png', dpi=150, bbox_inches='tight')
plt.show()
print('plot saved')

def load_set(folder):
    files = sorted(os.listdir(folder))
    data = []
    for f in files:
        if f.endswith('.txt'):
            file_path = os.path.join(folder, f)
            data.append(np.loadtxt(file_path))
    return np.array(data)

print("Loading all recordings...")
normal_all = load_set(os.path.join(data_path, "Z"))
seizure_all = load_set(os.path.join(data_path, "S"))

print(f"Normal set: {normal_all.shape}")
print(f"Seizure set: {seizure_all.shape}")