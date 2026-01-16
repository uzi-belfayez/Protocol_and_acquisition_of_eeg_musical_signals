# quick_check.py
import mne
import matplotlib.pyplot as plt
from pathlib import Path

# --- config ---
gdf_path = Path(r"C:\Users\rayen\eeg\signals\joel_second_two_ears.gdf")
sensors_map_path = Path(r"C:\Users\rayen\eeg\sensors_coordinates.txt")
marker_delay_s = 0.150 # marker was sent before note_on
# Limit to these note names, or set to None for all notes found
notes_include = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
reject_peak_to_peak_uv = 150  # set to None to disable artifact rejection

tmin, tmax = -0.2, 0.6

# --- load ---
raw = mne.io.read_raw_gdf(gdf_path, preload=False, verbose='error')
print("sfreq:", raw.info["sfreq"])
print("nchan:", raw.info["nchan"])
print("duration_s:", raw.n_times / raw.info["sfreq"])

# --- rename channels using the cap mapping (A1..An -> standard labels) ---
if sensors_map_path.exists():
    mapping = {}
    for line in sensors_map_path.read_text(encoding="ascii").splitlines():
        if "->" not in line:
            continue
        left, right = [s.strip() for s in line.split("->", 1)]
        if left.startswith("A") and left[1:].isdigit():
            idx = int(left[1:])
            mapping[f"Channel {idx}"] = right
    if mapping:
        raw.rename_channels(mapping)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="ignore")
        # Exclude unmapped channels from topomap-related plots
        unmapped = [
            ch for ch in raw.ch_names
            if ch.startswith("Channel ") or ch.startswith("EX ")
        ]
        if unmapped:
            raw.set_channel_types({ch: "misc" for ch in unmapped})

# --- basic filtering for visualization ---
raw_filt = raw.copy().load_data()
raw_filt.filter(1, 40, fir_design="firwin", verbose='error')
raw_filt.notch_filter([50, 100], verbose='error')  # change to 60/120 if needed

# --- find events (GDF annotations -> stim channel) ---
events, event_id = mne.events_from_annotations(raw_filt, verbose='error')
print("event_id:", event_id)
print("n_events:", len(events))
print("event_codes:", sorted(set(events[:, 2].tolist())))

# Build note label mapping from annotation keys (e.g., "33060")
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
def midi_to_note(midi_val: int) -> str:
    name = NOTE_NAMES[midi_val % 12]
    octave = (midi_val // 12) - 1
    return f"{name}{octave}"

note_event_id = {}
for key, code in event_id.items():
    if key.isdigit():
        stim_id = int(key)
        if stim_id >= 33000:
            midi_val = stim_id - 33000
            note_name = midi_to_note(midi_val)
            note_event_id[note_name] = code

if notes_include:
    note_event_id = {
        name: code for name, code in note_event_id.items()
        if name in notes_include
    }

if not note_event_id:
    raise RuntimeError(
        "No note events found. "
        f"Available event labels: {sorted(event_id.keys())}"
    )

print("note_event_id:", note_event_id)

# --- shift events by marker_delay_s ---
shift = int(marker_delay_s * raw_filt.info["sfreq"])
events_shifted = events.copy()
events_shifted[:, 0] += shift

# --- epochs + ERP ---
epochs = mne.Epochs(
    raw_filt,
    events_shifted,
    event_id=None,  # use all event types present in events
    tmin=tmin,
    tmax=tmax,
    baseline=(None, 0),
    event_repeated="merge",
    preload=True,
    verbose='error'
)

evoked = epochs.average()
evoked.plot(spatial_colors=True, time_unit='s', picks="eeg", show=False)

# --- quick PSD ---
raw_filt.compute_psd(fmax=60).plot(show=False)
raw.compute_psd(fmax=60).plot(show=False)
plt.show(block=True)
