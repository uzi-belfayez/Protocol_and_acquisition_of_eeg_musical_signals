# tonal_mappin.py
# Purpose: group EEG events into tonal categories and visualize long-window topomaps.
import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mne


def plot_band_topomaps(
    epochs_by_group,
    freq_band,
    band_name,
    fmin_psd=1,
    fmax_psd=80,
    n_fft=2048,
    cmap="RdBu_r"
):
    """
    Plot topomaps for one frequency band across 3 groups.

    Parameters
    ----------
    epochs_by_group : dict
        {"Major": Epochs, "Minor": Epochs, "Post_tonal": Epochs}
    freq_band : tuple
        (fmin, fmax) of the band
    band_name : str
        Name of the band (e.g. "Theta")
    """

    band_maps = {}

    # ----------------------------
    # Compute band power per group
    # ----------------------------
    for group, epochs in epochs_by_group.items():
        # Compute a PSD per epoch, then average within the band.
        psd = epochs.compute_psd(
            fmin=fmin_psd,
            fmax=fmax_psd,
            method="welch",
            n_fft=n_fft
        )

        freqs = psd.freqs
        psd_data = psd.get_data()  # (n_epochs, n_channels, n_freqs)

        fmin, fmax = freq_band
        mask = (freqs >= fmin) & (freqs <= fmax)

        band_power = psd_data[:, :, mask].mean(axis=2)
        band_power_db = 10 * np.log10(band_power)

        # moyenne inter-epochs topomap
        band_maps[group] = band_power_db.mean(axis=0)

    # ----------------------------
    # Common color scale so all groups are comparable.
    # ----------------------------
    all_vals = np.concatenate(list(band_maps.values()))
    vmax = np.max(np.abs(all_vals))
    vlim = (-vmax, vmax)

    # ----------------------------
    # Plot
    # ----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), constrained_layout=True)

    for ax, (group, data) in zip(axes, band_maps.items()):
        im, _ = mne.viz.plot_topomap(
            data,
            epochs.info,
            axes=ax,
            show=False,
            sensors=True,
            contours=0,
            cmap=cmap,
            vlim=vlim
        )
        ax.set_title(group)

    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Power (dB)")

    plt.suptitle(f"{band_name} band ({freq_band[0]}â€“{freq_band[1]} Hz)")
    plt.show()


print(mne.__version__)

# -----------------------------
# 1. config
# ----------------------------- 
# GDF recording to analyze and channel mapping to 10-20 labels.
gdf_path = Path(r"E:\ISN 3A\PRDA\signals\steven.gdf")
sensors_map_path = Path(r"C:\Users\rayen\eeg\sensors_coordinates.txt")
# marker_delay_s = 0.150 # marker was sent before note_on
# Limit to these note names, or set to None for all notes found
# notes_include = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
reject_peak_to_peak_uv = 150  # set to None to disable artifact rejection


# -----------------------------
# 2. load 
# -----------------------------
raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose='error')
print("sfreq:", raw.info["sfreq"])
print("nchan:", raw.info["nchan"])
print("duration_s:", raw.n_times / raw.info["sfreq"])

# --- rename channels using the cap mapping (A1..An -> standard labels) ---
if sensors_map_path.exists():
    # Build a dict like {"Channel 1": "Fp1", ...} from the mapping file.
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
# raw_filt = raw.copy().load_data()
raw.filter(1, 40, fir_design="firwin", verbose='error')
raw.notch_filter([50, 100], verbose='error')  # change to 60/120 if needed
raw.set_eeg_reference("average")

# Global PSD snapshot for quick inspection of spectral content.
psd = raw.compute_psd(fmin = 1, fmax=80, method="welch",n_fft=2048)
psd.plot(show=False)
plt.show(block=False)

# --- find events (GDF annotations -> stim channel) ---
events, event_id = mne.events_from_annotations(raw, verbose='error')
print("event_id:", event_id)
print("n_events:", len(events))
print("event_codes:", sorted(set(events[:, 2].tolist())))

# -----------------------------
# 3. DÃ©finition des groupes
# -----------------------------
# Assign each event label to a tonal group based on its last character.
group_map = {
    "1": "Major",
    "2": "Minor",
    "3": "Post_tonal"
}

# Explicit label lists (not currently used to rebuild epochs_groups).
group_labels = {
    "Major": [],
    "Minor": [],
    "Post_tonal": []
}
# Group labels by their last digit (e.g., ...1 => Major).
for lbl in event_id:

    key = lbl[-1]

    if key in group_map:
        group = group_map[key]
        group_labels[group].append(lbl)

# -----------------------------
# 4. Events & epochs
# -----------------------------
events, event_id = mne.events_from_annotations(raw)

# Build long epochs to capture slow responses (0-20s).
epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=-0.2,
    tmax=20,
    baseline=(-0.2, 0),
    preload=True,
    reject_by_annotation=True
)

epochs_groups = {}

# Concatenate epochs per tonal condition, then average.
for cond, labels in group_labels.items():
    epochs_groups[cond] = mne.concatenate_epochs(
        [epochs[lbl] for lbl in labels]
    )

evokeds = {
    cond: ep.average()
    for cond, ep in epochs_groups.items()
}

# -----------------------------
# 5. Definir fenetres ERP
# -----------------------------
#windows = {
#     "N1": (0.08, 0.12),
#     "P2": (0.15, 0.20),
#     "N2": (0.22, 0.35),
#     "LPP": (0.40, 0.80),
#     "begin": (1., 3.),
#     "middle": (3., 10.),
#     "end": (10., 20.)
# }

# Time windows (seconds) used for topomap averages.
windows = {
     "start": (0., 2.),
     "2": (2, 4),
     "3": (4,6),
     "4": (6,8),
     "5": (8,10),
     "6": (10,12),
     "7": (12,14),
     "8": (14,16),
     "9": (16,18),
     "end": (18,20)
 }

# ---------------------------
# 6.1. DÃ©finir les bandes de frÃ©quence
# ---------------------------
group_labels = {
    "Major": ["34002", "34003", "34006", "34007"],
    "Minor": ["34000", "34008", "34010", "34011"],
    "Post_tonal": ["34001", "34004", "34005", "34009"]
}

# freq_bands = {
#     "Theta": (4, 7),
#     'Alpha_low': (8, 10),
#     'Alpha_high': (10, 12),
#     'Beta_low': (13, 20),
#     'Beta_high': (20, 30),
#     'Gamma_low': (30, 50),
#     'Gamma_high': (50, 80)
# }



# band_topomaps = {group: {} for group in group_labels}

# # boucler sur les goupes pour toutes les bandes de frÃ©quences
# for group, ep in epochs_groups.items():

#     # PSD sur les epochs
#     psd = ep.compute_psd(
#         fmin=1,
#         fmax=80,
#         method="welch",
#         n_fft=2048
#     )
    
#     freqs = psd.freqs
#     psd_data = psd.get_data()
#     # boucle sur les bandes de frÃ©quence
#     for band, (fmin, fmax) in freq_bands.items():

#         mask = (freqs >= fmin) & (freqs <= fmax)

#         # moyenne frÃ©quence â†’ canal â†’ epoch
#         band_power = psd_data[:, :, mask].mean(axis=2)

#         # log-transform
#         band_power_db = 10 * np.log10(band_power)

#         # moyenne inter-epochs â†’ topomap
#         band_topomaps[group][band] = band_power_db.mean(axis=0)


# plot_band_topomaps(
#     epochs_by_group=epochs_groups,
#     freq_band=freq_bands["Beta_low"],
#     band_name="Beta_low"
# )

# -----------------------------
# 6. Calcul des topomaps
# -----------------------------
# -------------------------------------------------
# 1. Calcul d'une Ã©chelle commune (important)
# -------------------------------------------------
# Collect per-window channel means to set a shared color scale.
all_data = []

for cond in group_labels:
    evoked = evokeds[cond]
    for (tmin, tmax) in windows.values():
        crop = evoked.copy().crop(tmin=tmin, tmax=tmax)
        all_data.append(crop.data.mean(axis=1))

vmax = np.max(np.abs(np.concatenate(all_data)))
vmin = -vmax

# -------------------------------------------------
# 2. CrÃ©ation du canvas
# -------------------------------------------------
fig, axes = plt.subplots(
    nrows=len(group_labels),
    ncols=len(windows),
    figsize=(14, 9),
    constrained_layout=True
)

# -------------------------------------------------
# 3. Boucle d'affichage
# -------------------------------------------------
for i_cond, cond in enumerate(group_labels):
    evoked = evokeds[cond]

    for j_win, (win_name, (tmin, tmax)) in enumerate(windows.items()):

        ax = axes[i_cond, j_win]

        # Average the evoked response within the window to get one scalp map.
        evoked_crop = evoked.copy().crop(tmin=tmin, tmax=tmax)
        data = evoked_crop.data.mean(axis=1)

        im, _ = mne.viz.plot_topomap(
            data,
            evoked.info,
            axes=ax,
            show=False,
            contours=0,
            sensors=True,
            cmap="RdBu_r",
            vlim=(vmin, vmax)
        )

        # Titres
        if i_cond == 0:
            ax.set_title(win_name, fontsize=12)
        if j_win == 0:
            ax.set_ylabel(cond, fontsize=12)

# -------------------------------------------------
# 4. Colorbar unique
# -------------------------------------------------
cbar = fig.colorbar(
    im,
    ax=axes,
    orientation="vertical",
    fraction=0.02,
    pad=0.02
)
cbar.set_label("Amplitude (ÂµV)")

plt.show()
