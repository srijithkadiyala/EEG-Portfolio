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