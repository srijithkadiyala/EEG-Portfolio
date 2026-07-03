import numpy as np
import matplotlib.pyplot as plt
import mne
raw = mne.io.read_raw_edf('SC4001E0-PSG.edf', preload=True)
print(raw.info)
annotations = mne.read_annotations('SC4001EC-Hypnogram.edf')
raw.set_annotations(annotations)
print("Sleep stages loaded")
print(annotations)
eeg_channel= raw.pick_channels(['EEG Fpz-Cz'])
data, time = raw[0, :3000]
plt.figure(figsize=(2, 4))
plt.plot(time, data[0] * 1e6, color='navy')
plt.title('Real EEG Recording - First 30 seconds (1989)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (µV)')
plt.savefig('real_eeg_sample.png', dpi=150, bbox_inches='tight')
plt.show()