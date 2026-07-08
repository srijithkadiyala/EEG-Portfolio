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

def wavelet_energy (data, wavelet='db4' , level=4):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    energies = [np.sum(c**2) for c in coeffs]
    return energies
normal_wav = wavelet_energy(normal)
seizure_wav = wavelet_energy(seizure)

print (f"Normal wavelet energies: {[f'{e:.1f}' for e in normal_wav]}")
print (f"Seizure wavelet energies: {[f'{e:.1f}' for e in seizure_wav]}")

def load_set(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith('.txt')])
    return np.array([np.loadtxt(os.path.join(folder, f)) for f in files])

def extract_features(data):
    delta = bandpower (data, fs, 0.5, 4)
    alpha = bandpower(data, fs, 8, 13)
    beta = bandpower(data, fs, 13, 30)
    act, mob, com = hjorth(data)
    ent = ant.sample_entropy(data)
    wav = wavelet_energy(data)
    return [delta, alpha, beta, act, mob, com, ent] + wav

data_path = "/Users/sri/Projects/EEG-Signal-Filtering/EEG signal filtering/Project3_Epilepsy_Detection/Bonn University Dataset"
print("loading all 200 recordings...")
normal_all = load_set(os.path.join(data_path, "Z"))
seizure_all = load_set(os.path.join(data_path, "S"))

print(f"Normal set loaded: {normal_all.shape}")
print(f"Seizure set loaded: {seizure_all.shape}")

print("Extracting features...")
X = np.array([extract_features(r) for r in normal_all] +
            [extract_features(r) for r in seizure_all])
y = np.array([0]*100 + [1]*100)

print(f"Feature matrix shape: {X.shape}")
print(f"Labells: {sum(y==0)} normal, {sum(y==1)} seizure")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred= clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Accuracy with 12 features: {accuracy * 100:.1f}%")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Seizure']))

feature_names = ['Delta', 'Alpha', 'Beta', 
                 'Activity', 'Mobility', 'Complexity', 'Entropy',
                 'Wav1', 'Wav2', 'Wav3', 'Wav4', 'Wav5']

importance = clf.feature_importances_

plt.figure(figsize=(12, 5))
plt.bar(feature_names, importance, color='darkred')
plt.title('Feature Importance for Epilepsy Detection')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.savefig('epilepsy_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Project 4 complete")