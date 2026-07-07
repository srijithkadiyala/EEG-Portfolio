import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import antropy as ant
import pywt
import os

data_path = "/Users/sri/Projects/EEG-Signal-Filtering/EEG signal filtering/Project3_Epilepsy_Detection/Bonn University Dataset"

normal_file = os.path.join(data_path, "Z", "Z001.txt")
seizure_file = os.path.join(data_path, "S", "S001.txt")

normal = np.loadtxt(normal_file)
seizure = np.loadtxt(seizure_file)

print(f"First value of normal: {normal[0]}")
print(f"First value of seizure: {seizure[0]}")

fs = 173.6

print(f"Normal recording loaded: {len(normal)} samples")
print(f"Seizure recording loaded: {len(seizure)} samples")

def bandpower (data, fs, low, high):
    f, psd = signal.welch(data, fs, nperseg=256)
    mask = (f >= low) & (f <= high)
    return np.trapezoid(psd[mask], f[mask])

print(f"Normal max: {normal.max():.2f}")
print(f"Seizure max: {seizure.max():.2f}")

normal_delta = bandpower(normal, fs, 0.5, 4)
normal_alpha = bandpower(normal, fs, 8, 13)
normal_beta =  bandpower(normal, fs, 13, 30)

seizure_delta = bandpower(seizure, fs, 0.5, 4)
seizure_alpha = bandpower(seizure, fs, 8, 13)
seizure_beta = bandpower(seizure, fs, 13, 30)

print(f"Normal  - Delta: {normal_delta:.2f} | Alpha: {normal_alpha:.2f} | Beta: {normal_beta:.2f}")
print(f"Seizure - Delta: {seizure_delta:.2f} | Alpha: {seizure_alpha:.2f} | Beta: {seizure_beta:.2f}")

def hjorth(data):
    activity = np.var(data)
    mobility = np.sqrt(np.var(np.diff(data)) / np.var(data))
    complexity = np.sqrt(np.var(np.diff(np.diff(data))) / np.var(np.diff(data))) / mobility
    return activity , mobility, complexity

n_act, n_mob, n_com = hjorth(normal)
s_act, s_mob, s_com = hjorth(seizure)

print (f"Normal - Activity: {n_act:.2f} | Mobility: {n_mob:.2f} | Complexity: {n_com:.2f}")
print (f"Seizure - Activity: {s_act:.2f} | Mobility: {s_mob:.2f} | Complexity: {s_com:.2f}")

normal_entropy = ant.sample_entropy(normal)
seizure_entropy = ant.sample_entropy(seizure)

print(f"Normal entropy: {normal_entropy:.4f}")
print(f"Seizure entroopy: {seizure_entropy:.4f}")
