# EEG quick checks

This repo has two quick‑check scripts:

1) `quick_check.py`  
- Minimal single‑file check for `signals/joel_first_two_ears.gdf`.  
- Reads GDF, filters 1–40 Hz + notch 50/100 Hz, extracts events, shifts markers, plots ERP + PSD.  

2) `quick_check_notes.py`  
- Multi‑note analysis for `signals/joel_second_two_ears.gdf`.  
- Adds channel name mapping from `sensors_coordinates.txt` (A1.. -> P3..), applies a standard 10‑20 montage.  
- Marks unmapped `Channel *` and `EX *` as `misc` to avoid topomap errors.  
- Builds NOTE labels from stim codes (33000 + MIDI) and compares ERPs per note.  
- Plots topomaps at 0.1/0.2/0.3 s per note.  
- Shows PSD for filtered and raw data; all figures stay open until closed.

## Run

```powershell
uv run .\quick_check.py
uv run .\quick_check_notes.py
```

## WAV rating UI

- `wav_rating_app.py` plays shuffled .wav files from `extraits\Maj` and shows a 10s valence-arousal grid per file.
- Ratings are saved to `ratings.csv`.

```powershell
uv run .\wav_rating_app.py
```

## Notes

- `sensors_coordinates.txt` only maps A‑labels to 10‑20 positions; it is not numeric coordinates.  
- If you see line‑noise at 50 Hz in raw PSD, it is expected; the filtered PSD removes it.  
