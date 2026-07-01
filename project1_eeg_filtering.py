"""
=================================================================
PROJECT 1: EEG Signal Filtering using Python
PhD Application Portfolio — Srijith Kadiyala
Day 1–2 of 10
=================================================================

WHAT YOU WILL LEARN:
  • What EEG signals are and how to represent them in Python
  • FFT: Convert a time-domain signal to the frequency domain
  • Butterworth bandpass filters: Isolate specific brain wave bands
  • PSD: Measure signal power across frequencies
  • How to visualise everything with matplotlib

BRAIN WAVE FREQUENCY BANDS (core knowledge for this PhD!):
  ┌──────────┬────────────┬──────────────────────────────────┐
  │ Band     │ Freq (Hz)  │ Brain State                      │
  ├──────────┼────────────┼──────────────────────────────────┤
  │ Delta    │ 0.5 – 4    │ Deep sleep ← most relevant here  │
  │ Theta    │ 4 – 8      │ Light sleep / drowsiness         │
  │ Alpha    │ 8 – 13     │ Relaxed, eyes closed             │
  │ Beta     │ 13 – 30    │ Active thinking, alert           │
  │ Gamma    │ 30 – 100   │ Cognitive processing, memory     │
  └──────────┴────────────┴──────────────────────────────────┘

HOW TO RUN (in VS Code terminal — press Ctrl + ` to open it):
  pip install numpy scipy matplotlib
  python project1_eeg_filtering.py
=================================================================
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
# Each library has a specific job:
import numpy as np                  # Numerical arrays and maths
import matplotlib.pyplot as plt     # Plotting and visualisation
from scipy import signal            # Signal processing (filters, PSD)
from scipy.fft import fft, fftfreq  # Fast Fourier Transform

print("=" * 65)
print("  PROJECT 1: EEG Signal Filtering")
print("  PhD Portfolio – Srijith Kadiyala")
print("=" * 65)


# ════════════════════════════════════════════════════════════════
# PART 1 — BUILD A SYNTHETIC EEG SIGNAL
# ════════════════════════════════════════════════════════════════
#
# WHY SYNTHETIC? We do not need a real EEG file yet.
# We BUILD one by mixing frequency components — this teaches
# you exactly what EEG IS. Real EEG is just a mixture of all
# these waves at various strengths recorded from the scalp.
#

# Sampling parameters
fs       = 256    # Hz — 256 samples per second (standard clinical EEG)
duration = 10     # seconds of signal
t        = np.linspace(0, duration, duration * fs, endpoint=False)
# t is a 1-D array of 2560 equally spaced time points (0 to 9.996 seconds)

# Build individual brain wave components as sine waves
# Amplitude is in microvolts (μV) — realistic EEG amplitudes
delta_wave = 1.5 * np.sin(2 * np.pi *  2 * t)   # 2 Hz  — deep sleep
theta_wave = 1.0 * np.sin(2 * np.pi *  6 * t)   # 6 Hz  — light sleep
alpha_wave = 2.0 * np.sin(2 * np.pi * 10 * t)   # 10 Hz — relaxed (dominant)
beta_wave  = 0.5 * np.sin(2 * np.pi * 20 * t)   # 20 Hz — active thinking

# Add Gaussian noise (real EEG always has noise from muscles, electrodes etc.)
np.random.seed(42)    # Fix the seed so your results are reproducible every run
noise = 0.4 * np.random.randn(len(t))

# Combine everything — this is our simulated single-channel EEG signal
eeg_raw = delta_wave + theta_wave + alpha_wave + beta_wave + noise

print(f"\n[1/4] Synthetic EEG created")
print(f"      Duration={duration}s | fs={fs} Hz | Total samples={len(eeg_raw)}")


# ════════════════════════════════════════════════════════════════
# PART 2 — FFT: CONVERT TO FREQUENCY DOMAIN
# ════════════════════════════════════════════════════════════════
#
# FFT (Fast Fourier Transform):
#   Time domain  →  Frequency domain
#
# Analogy: shine white light through a prism and you see each
# individual colour (frequency). FFT does the same for a signal.
#
# The output shows WHICH frequencies exist and HOW STRONG they are.
# Peaks in the FFT tell you which brain waves are active.
#

N   = len(eeg_raw)
yf  = fft(eeg_raw)           # Complex frequency components (the "prism output")
xf  = fftfreq(N, d=1/fs)    # Frequency axis in Hz (d = sample spacing = 1/fs)

# Take only the positive frequency half (the other half is a mathematical mirror)
half  = N // 2
freqs = xf[:half]
amps  = (2.0 / N) * np.abs(yf[:half])   # Real amplitude at each frequency

print(f"[2/4] FFT computed")
print(f"      Frequency resolution: {freqs[1]:.4f} Hz per bin")


# ════════════════════════════════════════════════════════════════
# PART 3 — BUTTERWORTH FILTERS: ISOLATE EACH BRAIN WAVE BAND
# ════════════════════════════════════════════════════════════════
#
# After FFT tells us WHICH frequencies exist, a bandpass filter
# EXTRACTS only the frequencies we want and blocks the rest.
#
# Why Butterworth?
#   ✓ Maximally flat in the passband — does not distort kept signals
#   ✓ Smooth rolloff — gradually rejects out-of-band frequencies
#   ✓ Standard choice in clinical EEG / biomedical signal processing
#
# filtfilt() applies the filter forward AND backward, which
# cancels any phase shift — very important for EEG analysis.
#

def bandpass(data, low_hz, high_hz, sampling_rate, order=4):
    """
    Apply a zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    data          : 1-D numpy array — the raw EEG signal
    low_hz        : float — lower frequency cutoff (Hz)
    high_hz       : float — upper frequency cutoff (Hz)
    sampling_rate : int   — samples per second (Hz)
    order         : int   — filter order (4 is standard for EEG)

    Returns
    -------
    1-D numpy array — the filtered signal
    """
    nyquist = sampling_rate / 2.0       # Max detectable frequency (Nyquist theorem)
    low     = low_hz  / nyquist          # Normalise to 0–1 range
    high    = high_hz / nyquist
    b, a    = signal.butter(order, [low, high], btype='bandpass')  # Filter coefficients
    return  signal.filtfilt(b, a, data)  # Zero-phase filtering (forward + backward)


delta_f = bandpass(eeg_raw, 0.5,  4, fs)   # Deep sleep
theta_f = bandpass(eeg_raw,   4,  8, fs)   # Light sleep
alpha_f = bandpass(eeg_raw,   8, 13, fs)   # Relaxed wakefulness
beta_f  = bandpass(eeg_raw,  13, 30, fs)   # Active thinking

print(f"[3/4] Butterworth bandpass filters applied")
print(f"      Extracted bands: Delta (δ), Theta (θ), Alpha (α), Beta (β)")


# ════════════════════════════════════════════════════════════════
# PART 4 — POWER SPECTRAL DENSITY (PSD)
# ════════════════════════════════════════════════════════════════
#
# FFT gives the spectrum at ONE snapshot.
# PSD (Welch's method) averages many overlapping FFT windows
# to produce a STABLE estimate of how much POWER exists at each
# frequency over the entire recording.
#
# Unit: μV²/Hz — power per unit frequency.
# PSD is used in sleep staging and neurodevelopmental research.
# (You will use this extensively in Project 4!)
#

f_psd, psd = signal.welch(eeg_raw, fs, nperseg=256, noverlap=128)
# nperseg=256 means each FFT window is 256 samples = 1 second at fs=256
# noverlap=128 means 50% overlap between consecutive windows

print(f"[4/4] Power Spectral Density computed via Welch's method")


# ════════════════════════════════════════════════════════════════
# VISUALISATION — 7 panels
# ════════════════════════════════════════════════════════════════

colours = {
    'raw':   '#1a237e',   # Dark blue
    'fft':   '#1b5e20',   # Dark green
    'delta': '#0d47a1',   # Blue        — slow, deep sleep
    'theta': '#6a1b9a',   # Purple      — light sleep
    'alpha': '#e65100',   # Orange      — relaxed
    'beta':  '#b71c1c',   # Red         — active
    'psd':   '#37474f',   # Dark grey
}

# Vertical dashed lines showing band boundaries in spectrum panels
band_boundaries = [
    (4,  '4 Hz\n(δ/θ)'),
    (8,  '8 Hz\n(θ/α)'),
    (13, '13 Hz\n(α/β)'),
    (30, '30 Hz\n(β end)'),
]

fig, axes = plt.subplots(7, 1, figsize=(14, 22))
fig.patch.set_facecolor('#fafafa')
fig.suptitle(
    'Project 1: EEG Signal Filtering\n'
    'Srijith Kadiyala — PhD Portfolio',
    fontsize=16, fontweight='bold', color='#1a1a2e', y=0.998
)

n_plot = 3 * fs   # Plot only the first 3 seconds for clarity


# ── Panel 1: Raw mixed EEG signal ───────────────────────────
axes[0].plot(t[:n_plot], eeg_raw[:n_plot], color=colours['raw'], lw=0.9)
axes[0].set_title('① Raw EEG Signal — all brain waves mixed together + noise',
                   fontweight='bold', color=colours['raw'])
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude (μV)')
axes[0].set_xlim(0, 3)
axes[0].grid(True, alpha=0.25)


# ── Panel 2: FFT frequency spectrum ─────────────────────────
freq_mask = freqs <= 40   # Show up to 40 Hz
axes[1].plot(freqs[freq_mask], amps[freq_mask],
             color=colours['fft'], lw=1.5)
axes[1].set_title('② FFT Spectrum — peaks reveal which frequency components are present',
                   fontweight='bold', color=colours['fft'])
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Amplitude')
axes[1].set_xlim(0, 40)
axes[1].grid(True, alpha=0.25)
# Add band boundary markers
for hz, lbl in band_boundaries:
    axes[1].axvline(x=hz, color='grey', ls='--', lw=1.0, alpha=0.7)
    axes[1].text(hz + 0.3, amps[freq_mask].max() * 0.80,
                 lbl, fontsize=7.5, color='#555555')


# ── Panels 3–6: Individual filtered brain waves ──────────────
band_info = [
    (delta_f, '③ Delta (0.5–4 Hz) — Deep Sleep',          colours['delta']),
    (theta_f, '④ Theta (4–8 Hz) — Light Sleep / Drowsy',  colours['theta']),
    (alpha_f, '⑤ Alpha (8–13 Hz) — Relaxed Wakefulness',  colours['alpha']),
    (beta_f,  '⑥ Beta (13–30 Hz) — Active Thinking',       colours['beta']),
]

for i, (band_data, title, col) in enumerate(band_info):
    ax = axes[i + 2]
    ax.plot(t[:n_plot], band_data[:n_plot], color=col, lw=0.9)
    ax.set_title(title, fontweight='bold', color=col)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (μV)')
    ax.set_xlim(0, 3)
    ax.grid(True, alpha=0.25)


# ── Panel 7: Power Spectral Density ─────────────────────────
psd_mask = f_psd <= 40
axes[6].semilogy(f_psd[psd_mask], psd[psd_mask],
                 color=colours['psd'], lw=1.5)
axes[6].set_title('⑦ Power Spectral Density (PSD) — stable power per frequency | '
                   'Preview of Project 4',
                   fontweight='bold', color=colours['psd'])
axes[6].set_xlabel('Frequency (Hz)')
axes[6].set_ylabel('Power (μV²/Hz)')
axes[6].set_xlim(0, 40)
axes[6].grid(True, alpha=0.25)
for hz, lbl in band_boundaries:
    axes[6].axvline(x=hz, color='grey', ls='--', lw=1.0, alpha=0.7)


# ── Style all panels ─────────────────────────────────────────
for ax in axes:
    ax.set_facecolor('#ffffff')
    for spine in ax.spines.values():
        spine.set_color('#cccccc')

plt.tight_layout(rect=[0, 0, 1, 0.997])

output_path = 'project1_eeg_filtering.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#fafafa')
plt.show()


# ════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  PROJECT 1 COMPLETE!")
print("=" * 65)
print(f"  Output saved → {output_path}")
print()
print("  Concepts you just demonstrated:")
print("  [FFT]         Converts time-domain signal to frequency domain")
print("  [Butterworth] Bandpass filter isolates delta, theta, alpha, beta")
print("  [PSD]         Stable measure of power at each frequency band")
print()
print("  ── DAY 2 EXTENSION TASKS (to deepen this project) ──────")
print()
print("  1. Load a real EEG file (.edf format) — install MNE first:")
print("       pip install mne")
print("       import mne")
print("       raw = mne.io.read_raw_edf('your_file.edf', preload=True)")
print()
print("  2. Add a 50 Hz NOTCH FILTER (removes mains power line noise")
print("     from European recordings — very important in real EEG):")
print("       b, a = signal.iirnotch(50, 30, fs)")
print("       eeg_clean = signal.filtfilt(b, a, eeg_raw)")
print()
print("  3. Plot a SPECTROGRAM (shows how frequency content changes")
print("     over time — key for sleep stage analysis):")
print("       f, t_spec, Sxx = signal.spectrogram(eeg_raw, fs)")
print("       plt.pcolormesh(t_spec, f[:50], 10*np.log10(Sxx[:50]))")
print()
print("  4. Try different filter orders (2, 4, 6, 8) and compare the")
print("     sharpness of the frequency cutoff in the filtered signals.")
print()
print("  Ready for Day 3? Move to: project2_sleep_setup.py")
print("=" * 65)
