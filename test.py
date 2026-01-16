import mne
from pathlib import Path
raw = mne.io.read_raw_gdf(r"C:\Users\rayen\eeg\signals\joel_first_two_ears.gdf", preload=False, verbose="error")
print(raw.ch_names[:40])