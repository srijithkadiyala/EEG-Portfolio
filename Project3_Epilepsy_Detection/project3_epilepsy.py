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

normal_labels = np.zeros(100)
seizure_labels = np.ones(100)

X = np.vstack((normal_all, seizure_all))
y = np.concatenate((normal_labels, seizure_labels))

print(f"Dataset ready: {X.shape[0]}recordings, {X.shape[1]} features each")
print(f"Labels: {int(sum(y==0))} normal, {int(sum(y==1))} seizure")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test,y_train,  y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit (X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"accuracy: {accuracy * 100:.1f}%")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Seizure']))

import matplotlib.pyplot as plt

feature_importance = clf.feature_importances_
plt.figure(figsize=(12, 4))
plt.plot(feature_importance, color='darkred')
plt.title('Feature Importance - Seizure detection')
plt.xlabel('Time point')
plt.ylabel('Importane')
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Feature Importance plot saved")

top_feature =np.argmax(feature_importance)
print(f"Most mportant time point: sample{top_feature}")
print(f"That is {top_feature/173.6:.2f} seconds into recording")

fig, axes = plt.subplots(2, 1, figsize=(12,6))

axes[0].plot(t, normal, color='blue')
axes[0].axvline(x=7.24, color='red', linestyle='--', linewidth=2, label='Key moment')
axes[0].set_title('Normal brain activity - Key moment at 7.24 sec')
axes[0].set_xlabel('Time(seconds)')
axes[0].set_ylabel('Amplitude(Uv)')
axes[0].legend()

axes[1].plot(t, seizure, color='red')
axes[1].axvline(x=7.24, color='black', linestyle='--', linewidth=2, label='KEy moment')
axes[1].set_title('Seizure brain activity - Key moment at 7.24 sec')
axes[1].set_xlabel('Time(seconds)')
axes[1].set_ylabel('Amplitude(Uv)')
axes[1].legend()

plt.tight_layout()
plt.savefig('normal_vs_seizure_annotated.png', dpi=150, bbox_inches='tight')
plt.show()
print("Annotated plot saved")

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)

y_pred_svm = svm.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print(f"Random Forest ccuracy = 97.5%")
print(f"SVM accuracy :{accuracy_svm * 100:.1f}%")