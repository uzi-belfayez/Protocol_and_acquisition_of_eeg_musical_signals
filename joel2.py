# tonal_mappin.py
import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
from scipy.stats import zscore

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import mne

def fix_nans(raw, bad_ratio=0.01, verbose=True):
    """
    Nettoie les NaN / Inf dans un objet mne.Raw.

    ParamÃ¨tres
    ----------
    raw : mne.io.Raw
        DonnÃ©es EEG chargÃ©es (raw.load_data()).
    bad_ratio : float
        Proportion max tolÃ©rÃ©e de NaN avant de marquer un canal bad (ex: 0.01 = 1%).
    verbose : bool
        Affiche les infos si True.

    Retour
    ------
    raw : mne.io.Raw
        Objet raw nettoyÃ© (modifiÃ© inplace).
    """

    if not raw.preload:
        raise RuntimeError("raw must be preloaded (raw.load_data())")

    data = raw.get_data()

    n_ch, n_times = data.shape

    # DÃ©tection NaN / Inf
    bad_mask = np.isnan(data) | np.isinf(data)

    nan_counts = bad_mask.sum(axis=1)

    threshold = bad_ratio * n_times

    bad_channels = [
        raw.ch_names[i]
        for i, n in enumerate(nan_counts)
        if n > threshold
    ]

    if verbose:
        print("---- fix_nans report ----")
        print(f"Samples par canal : {n_times}")
        print(f"Seuil NaN : {int(threshold)} ({bad_ratio*100:.2f}%)")

        if bad_channels:
            print("Canaux marquÃ©s bad :", bad_channels)
        else:
            print("Aucun canal fortement corrompu")

    # Marquage bad channels
    raw.info["bads"].extend(bad_channels)

    # Interpolation temporelle des petits trous
    data_clean = data.copy()

    for i in range(n_ch):

        y = data_clean[i]

        bad = np.isnan(y) | np.isinf(y)

        if not bad.any():
            continue

        # Si canal dÃ©jÃ  bad â†’ skip
        if raw.ch_names[i] in bad_channels:
            continue

        good = ~bad

        # SÃ©curitÃ© : au moins 2 points valides
        if good.sum() < 2:
            raw.info["bads"].append(raw.ch_names[i])
            continue

        f = interp1d(
            np.where(good)[0],
            y[good],
            bounds_error=False,
            fill_value="extrapolate"
        )

        y[bad] = f(np.where(bad)[0])

        data_clean[i] = y

        if verbose:
            print(f"Interpolation temporelle : {raw.ch_names[i]} ({bad.sum()} pts)")

    # RÃ©injection dans raw
    raw._data = data_clean

    if verbose:
        print("---- fin fix_nans ----\n")

    return raw


plot_signals = True
plot_duration_s = 10.0
plot_start_s = 0.0
plot_max_channels = 20
apply_plot_filter = True
plot_lfreq = 1.0
plot_hfreq = 40.0
plot_notch_freqs = [50, 100]
epoch_decim = 8  # downsample epochs to reduce memory (2048 Hz -> 256 Hz)

print(mne.__version__)

# -----------------------------
# 1. config
# ----------------------------- 
gdf_path = Path(r"E:\ISN 3A\PRDA\signals\habbi_interface_2.gdf")
sensors_map_path = Path(r"C:\Users\rayen\eeg\sensors_coordinates.txt")

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
# raw.set_eeg_reference("average")

psd = raw.compute_psd(fmin = 1, fmax=80, method="welch",n_fft=2048)
psd.plot(show=False)
plt.show(block=False)

# --- find events (GDF annotations -> stim channel) ---
events, event_id = mne.events_from_annotations(raw, verbose='error')
print("event_id:", event_id)
print("n_events:", len(events))
print("event_codes:", sorted(set(events[:, 2].tolist())))

# dÃ©tection des NaN Ã  l'acquisition + interpolation temporelle pour correction
raw_plot = fix_nans(raw)

# recherche de canaux aberrants 
psd = raw_plot.compute_psd(
    fmin=1,
    fmax=80,
    method="welch",
    n_fft=2048,
    picks="eeg",
    exclude="bads"
)
psd_data = psd.get_data()
freqs = psd.freqs
mean_power = psd_data.mean(axis=1)
log_power = np.log10(mean_power)   # toujours mieux en log
# z test pour dÃ©terminer les outliers
z = zscore(log_power)
bad_idx = np.where(np.abs(z) > 3)[0]   # seuil classique
bad_chs = [raw_plot.ch_names[i] for i in bad_idx]

print("Canaux aberrants :", bad_chs)
raw_plot.info["bads"].extend(bad_chs)
# interpolation pour correction du canal aberrant
raw_plot.interpolate_bads(reset_bads=True)
# nouveau contrÃ´le des NaN
raw_plot = fix_nans(raw_plot)
data = raw_plot.get_data()

print("NaN restants :", np.isnan(data).sum())
print("Inf restants :", np.isinf(data).sum())

# calcul des spectres finaux
raw_plot.set_eeg_reference("average")


psd = raw_plot.compute_psd(fmin = 1, fmax=80, method="welch",n_fft=2048)
psd_data = psd.get_data()
freqs = psd.freqs
mean_power = psd_data.mean(axis=1)
log_power = np.log10(mean_power)   # toujours mieux en log

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
group_map = {
    "1": "Major",
    "2": "Minor",
    "3": "Post_tonal"
}

group_labels = {
    "Major": [],
    "Minor": [],
    "Post_tonal": []
}
for lbl in event_id:

    key = lbl[-1]

    if key in group_map:
        group = group_map[key]
        group_labels[group].append(lbl)

# group_labels = {
#     "Major": ["34002", "34003", "34006", "34007"],
#     "Minor": ["34000", "34008", "34010", "34011"],
#     "Post_tonal": ["34001", "34004", "34005", "34009"]
# }

for group, labels in group_labels.items():

    filtered = [lbl for lbl in labels if lbl.startswith("34")]

    group_labels[group] = filtered

# -----------------------------
# 4. Events & epochs
# -----------------------------
events, event_id = mne.events_from_annotations(raw)

epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=-0.2,
    tmax=20,
    baseline=(-0.2, 0),
    decim=epoch_decim,
    preload=False,
    reject_by_annotation=True
)

# Average per group directly to avoid concatenating large epoch arrays.
evokeds = {
    cond: epochs[labels].average()
    for cond, labels in group_labels.items()
}

# -----------------------------
# 5. DÃ©finir fenÃªtres ERP
# -----------------------------
# windows = {
#     "N1": (0.08, 0.12),
#     "P2": (0.15, 0.20),
#     "N2": (0.22, 0.35),
#     "LPP": (0.40, 0.80),
#     "begin": (1., 3.),
#     "middle": (3., 10.),
#     "end": (10., 20.)
# }

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

for group, labels in group_labels.items():
    print(f"{group}: {', '.join(labels)}")

# -----------------------------
# 6. Calcul des topomaps
# -----------------------------
# -------------------------------------------------
# 1. Calcul d'une Ã©chelle commune (important)
# -------------------------------------------------
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
