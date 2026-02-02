import mne
import numpy as np
from pathlib import Path

# --- config ---
gdf_path = Path(r"C:\Users\rayen\eeg\signals\joel_second_two_ears.gdf")
sensors_map_path = Path(r"C:\Users\rayen\eeg\sensors_coordinates.txt")
marker_delay_s = 0.150  # marker was sent before note_on
notes_include = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
reject_peak_to_peak_uv = 150  # set to None to disable artifact rejection
tmin, tmax = -0.2, 0.6

# ROI where auditory ERPs are commonly strong
roi_channels = ["FC3", "FC4", "C3", "C4", "CP3", "CP4", "F3", "F4"]

# Spectral settings
bands = {
    "alpha": (8, 12),
    "beta": (13, 30),
    "gamma": (30, 45),
}
baseline_win = (tmin, 0.0)
post_win = (0.0, 0.5)
psd_fmin = 4.0
psd_fmax = 45.0
eps = 1e-20

# --- load ---
raw = mne.io.read_raw_gdf(gdf_path, preload=False, verbose="error")
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
        unmapped = [
            ch for ch in raw.ch_names
            if ch.startswith("Channel ") or ch.startswith("EX ")
        ]
        if unmapped:
            raw.set_channel_types({ch: "misc" for ch in unmapped})

# --- filtering for spectral analysis ---
raw_spec = raw.copy().load_data()
raw_spec.notch_filter([50, 100], verbose="error")
raw_spec.filter(1, 80, fir_design="firwin", verbose="error")

# --- find events (GDF annotations -> stim channel) ---
events, event_id = mne.events_from_annotations(raw_spec, verbose="error")
print("event_id:", event_id)
print("n_events:", len(events))
print("event_codes:", sorted(set(events[:, 2].tolist())))

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

# --- shift events by marker_delay_s ---
shift = int(marker_delay_s * raw_spec.info["sfreq"])
events_shifted = events.copy()
events_shifted[:, 0] += shift

reject = None
if reject_peak_to_peak_uv:
    reject = {"eeg": reject_peak_to_peak_uv * 1e-6}

epochs = mne.Epochs(
    raw_spec,
    events_shifted,
    event_id=note_event_id,
    tmin=tmin,
    tmax=tmax,
    baseline=None,
    event_repeated="merge",
    reject=reject,
    preload=True,
    verbose="error"
)

roi_picks = [ch for ch in roi_channels if ch in epochs.ch_names]
if not roi_picks:
    roi_picks = mne.pick_types(epochs.info, eeg=True, exclude="bads").tolist()
    roi_label = "all_eeg"
else:
    roi_label = ",".join(roi_picks)

def _trapezoid_integrate(y, x, axis=-1):
    dx = np.diff(x)
    if dx.size == 0:
        return np.nan
    y1 = np.take(y, indices=range(1, len(x)), axis=axis)
    y0 = np.take(y, indices=range(0, len(x) - 1), axis=axis)
    avg = (y1 + y0) * 0.5
    shape = [1] * avg.ndim
    shape[axis] = dx.shape[0]
    return np.sum(avg * dx.reshape(shape), axis=axis)


def _move_freq_axis(data, freqs_len):
    axes = [i for i, size in enumerate(data.shape) if size == freqs_len]
    if not axes:
        return None
    freq_axis = axes[-1]
    if freq_axis != data.ndim - 1:
        data = np.moveaxis(data, freq_axis, -1)
    return data


def bandpower_mean(psd_data, freqs, fmin, fmax):
    idx = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(idx):
        return float("nan")
    data = _move_freq_axis(psd_data, len(freqs))
    if data is None:
        raise RuntimeError(
            f"Could not align PSD data to freqs. data shape={psd_data.shape}, "
            f"freqs={len(freqs)}"
        )
    data = data[..., idx]
    freq_band = freqs[idx]
    if hasattr(np, "trapezoid"):
        band = np.trapezoid(data, freq_band, axis=-1)
    elif hasattr(np, "trapz"):
        band = np.trapz(data, freq_band, axis=-1)
    else:
        band = _trapezoid_integrate(data, freq_band, axis=-1)
    return band.mean()

print(f"\nBandpower dB change (ROI: {roi_label})")
print(f"baseline: {baseline_win[0]}..{baseline_win[1]} s | post: {post_win[0]}..{post_win[1]} s")
print(f"{'Note':<4} {'N':>3} {'Band':<5} {'dB':>8}")

for note in sorted(note_event_id.keys()):
    ep = epochs[note]
    n_trials = len(ep)
    if n_trials == 0:
        continue
    psd_pre = ep.compute_psd(
        method="welch",
        fmin=psd_fmin,
        fmax=psd_fmax,
        tmin=baseline_win[0],
        tmax=baseline_win[1],
        verbose="error"
    )
    psd_post = ep.compute_psd(
        method="welch",
        fmin=psd_fmin,
        fmax=psd_fmax,
        tmin=post_win[0],
        tmax=post_win[1],
        verbose="error"
    )
    freqs_pre = psd_pre.freqs
    freqs_post = psd_post.freqs
    pre_data = psd_pre.get_data(picks=roi_picks)
    post_data = psd_post.get_data(picks=roi_picks)
    for band, (bmin, bmax) in bands.items():
        pre = bandpower_mean(pre_data, freqs_pre, bmin, bmax)
        post = bandpower_mean(post_data, freqs_post, bmin, bmax)
        db = 10.0 * np.log10((post + eps) / (pre + eps))
        print(f"{note:<4} {n_trials:>3} {band:<5} {db:>8.2f}")
