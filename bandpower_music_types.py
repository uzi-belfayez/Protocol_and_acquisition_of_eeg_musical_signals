# Compute alpha/beta/gamma bandpower per music type (Major/Minor/Post-tonal).
# Uses start markers (34xxx) and fixed 0..20s windows.
import csv
from pathlib import Path

import numpy as np
import mne
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
GDF_PATH = Path(r"E:\ISN 3A\PRDA\signals\mohamed_1.gdf")
SENSORS_MAP_PATH = Path(r"C:\Users\rayen\eeg\sensors_coordinates.txt")

# Use start markers only (34xxx). Each segment is 0..20s.
SEGMENT_TMIN = 0.0
SEGMENT_TMAX = 20.0

# Downsample to reduce memory (2048 Hz -> 512 Hz).
TARGET_SFREQ = 512.0

# Filtering + PSD
FILTER_LFREQ = 1.0
FILTER_HFREQ = 50.0  # include gamma up to 47 Hz
NOTCH_FREQS = [50, 100]
FMIN_PSD = 1.0
FMAX_PSD = 47.0

# Reference (set False to keep original reference).
APPLY_AVG_REF = True

# Band definitions (Hz)
BANDS = {
    "theta": (3.0, 7.0),
    "alpha": (8.0, 13.0),
    "beta": (14.0, 29.0),
    "gamma": (30.0, 47.0),
}

# Output files
OUTPUT_LONG = Path("bandpower_by_epoch_channel.csv")
OUTPUT_SUMMARY = Path("bandpower_summary.csv")

# Z-score threshold for bad channels (PSD outliers)
BAD_Z_THRESHOLD = 3.0


def apply_channel_mapping(raw, map_path):
    if not map_path.exists():
        return
    mapping = {}
    for line in map_path.read_text(encoding="ascii").splitlines():
        if "->" not in line:
            continue
        left, right = [s.strip() for s in line.split("->", 1)]
        if left.startswith("A") and left[1:].isdigit():
            idx = int(left[1:])
            mapping[f"Channel {idx}"] = right
    if not mapping:
        return
    raw.rename_channels(mapping)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="ignore")
    unmapped = [ch for ch in raw.ch_names if ch.startswith("Channel ") or ch.startswith("EX ")]
    if unmapped:
        raw.set_channel_types({ch: "misc" for ch in unmapped})


def detect_bad_channels_psd(raw):
    psd = raw.compute_psd(
        fmin=FMIN_PSD,
        fmax=FMAX_PSD,
        method="welch",
        picks="eeg",
    )
    data = psd.get_data()
    mean_power = data.mean(axis=-1)
    eps = np.finfo(float).eps
    log_power = np.log10(mean_power + eps)
    z = (log_power - log_power.mean()) / (log_power.std(ddof=0) + eps)
    bad_idx = np.where(np.abs(z) > BAD_Z_THRESHOLD)[0]
    bad_chs = [psd.ch_names[i] for i in bad_idx]
    return bad_chs


def main():
    print(mne.__version__)

    raw = mne.io.read_raw_gdf(GDF_PATH, preload=True, verbose="error")
    print(f"sfreq: {raw.info['sfreq']}")
    print(f"nchan: {raw.info['nchan']}")
    print(f"duration_s: {raw.n_times / raw.info['sfreq']}")

    apply_channel_mapping(raw, SENSORS_MAP_PATH)

    # Filtering for PSD/bandpower
    raw.filter(FILTER_LFREQ, FILTER_HFREQ, fir_design="firwin", verbose="error")
    raw.notch_filter(NOTCH_FREQS, verbose="error")
    if APPLY_AVG_REF:
        raw.set_eeg_reference("average", verbose="error")

    # Find events (annotations -> events)
    events, event_id = mne.events_from_annotations(raw, verbose="error")

    # Group labels by last digit (1=Major, 2=Minor, 3=Post_tonal)
    group_map = {"1": "Major", "2": "Minor", "3": "Post_tonal"}
    group_labels = {name: [] for name in group_map.values()}
    for lbl in event_id:
        if not (lbl.isdigit() and lbl.startswith("34")):
            continue
        key = lbl[-1]
        if key in group_map:
            group_labels[group_map[key]].append(lbl)

    for group, labels in group_labels.items():
        print(f"{group}: {len(labels)} labels")

    # Detect bad channels using PSD z-score
    bad_chs = detect_bad_channels_psd(raw)
    if bad_chs:
        print(f"Bad channels (PSD z>{BAD_Z_THRESHOLD}): {', '.join(bad_chs)}")
    else:
        print("Bad channels (PSD z-score): none")
    raw.info["bads"].extend(bad_chs)

    # Decimate to reduce memory
    sfreq = raw.info["sfreq"]
    decim = max(1, int(round(sfreq / TARGET_SFREQ)))
    print(f"Decim: {decim} (target {TARGET_SFREQ} Hz -> {sfreq / decim:.1f} Hz)")

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=SEGMENT_TMIN,
        tmax=SEGMENT_TMAX,
        baseline=None,
        decim=decim,
        preload=False,
        reject_by_annotation=True,
        verbose="error",
    )

    # Prepare CSV writers
    OUTPUT_LONG.write_text(
        "group,event_label,event_code,epoch_index,event_sample,event_time_s,"
        "channel,is_bad,band,power,db\n",
        encoding="ascii",
    )
    with OUTPUT_LONG.open("a", newline="", encoding="ascii") as f_long, OUTPUT_SUMMARY.open(
        "w", newline="", encoding="ascii"
    ) as f_sum:
        long_writer = csv.writer(f_long)
        sum_writer = csv.writer(f_sum)
        sum_writer.writerow(
            [
                "group",
                "band",
                "n_epochs",
                "n_channels_used",
                "mean_power",
                "mean_db",
            ]
        )

        for group, labels in group_labels.items():
            if not labels:
                continue
            ep = epochs[labels]
            n_epochs = len(ep.events)
            if n_epochs == 0:
                continue

            picks_all = mne.pick_types(ep.info, eeg=True, exclude=())
            picks_good = mne.pick_types(ep.info, eeg=True, exclude="bads")
            psd = ep.compute_psd(
                method="welch",
                fmin=FMIN_PSD,
                fmax=FMAX_PSD,
                picks=picks_good,  # compute on good channels only
                exclude="bads",
                verbose="error",
            )
            psd_data = psd.get_data()  # (n_epochs, n_good_channels, n_freqs)
            freqs = psd.freqs
            ch_names = [ep.ch_names[i] for i in picks_all]
            # Expand PSD to include bad channels (filled with NaN) for the detailed CSV.
            index_map = {pick: idx for idx, pick in enumerate(picks_all)}
            full_psd = np.full(
                (psd_data.shape[0], len(picks_all), psd_data.shape[2]),
                np.nan,
                dtype=psd_data.dtype,
            )
            for j, pick in enumerate(picks_good):
                full_psd[:, index_map[pick], :] = psd_data[:, j, :]
            eps = np.finfo(float).eps

            code_to_label = {code: label for label, code in ep.event_id.items()}
            event_samples = ep.events[:, 0]
            event_times_s = event_samples / sfreq

            band_powers = {}
            band_powers_db = {}
            for band, (bmin, bmax) in BANDS.items():
                mask = (freqs >= bmin) & (freqs <= bmax)
                power = full_psd[..., mask].mean(axis=-1)
                band_powers[band] = power
                band_powers_db[band] = 10.0 * np.log10(power + eps)

            # Detailed CSV (includes bad channels)
            bad_set = set(raw.info["bads"])
            for ep_idx, code in enumerate(ep.events[:, 2]):
                label = code_to_label.get(code, "")
                for ch_idx, ch_name in enumerate(ch_names):
                    is_bad = "yes" if ch_name in bad_set else "no"
                    for band in BANDS:
                        power = band_powers[band][ep_idx, ch_idx]
                        power_db = band_powers_db[band][ep_idx, ch_idx]
                        long_writer.writerow(
                            [
                                group,
                                label,
                                code,
                                ep_idx,
                                int(event_samples[ep_idx]),
                                f"{event_times_s[ep_idx]:.3f}",
                                ch_name,
                                is_bad,
                                band,
                                f"{power:.6e}",
                                f"{power_db:.3f}",
                            ]
                        )

            # Summary CSV (exclude bad channels)
            good_mask = np.array([ch not in bad_set for ch in ch_names])
            for band in BANDS:
                power = band_powers[band][:, good_mask]
                power_db = band_powers_db[band][:, good_mask]
                mean_power = float(np.nanmean(power))
                mean_db = float(np.nanmean(power_db))
                sum_writer.writerow(
                    [
                        group,
                        band,
                        n_epochs,
                        int(good_mask.sum()),
                        f"{mean_power:.6e}",
                        f"{mean_db:.3f}",
                    ]
                )

    # -----------------------------
    # Plots
    # -----------------------------
    summary = np.genfromtxt(OUTPUT_SUMMARY, delimiter=",", names=True, dtype=None, encoding="ascii")

    groups = ["Major", "Minor", "Post_tonal"]
    bands = list(BANDS.keys())

    # 1) Grouped bar chart (bands on x-axis, groups as bars)
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    x = np.arange(len(bands))
    width = 0.25
    for i, group in enumerate(groups):
        vals = []
        for band in bands:
            row = summary[(summary["group"] == group) & (summary["band"] == band)]
            vals.append(row["mean_db"][0] if len(row) else np.nan)
        ax1.bar(x + i * width - width, vals, width, label=group)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bands)
    ax1.set_ylabel("Mean power (dB)")
    ax1.set_title("Bandpower by group (mean dB)")
    ax1.legend()

    # 2) Box plot (distributions across channels+epochs, exclude bads)
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    box_data = []
    box_labels = []
    for band in bands:
        for group in groups:
            rows = summary[(summary["group"] == group) & (summary["band"] == band)]
            if len(rows) == 0:
                continue
            # Re-load long CSV for distributions (exclude bads)
            long = np.genfromtxt(OUTPUT_LONG, delimiter=",", names=True, dtype=None, encoding="ascii")
            mask = (
                (long["group"] == group)
                & (long["band"] == band)
                & (long["is_bad"] == "no")
            )
            vals = long["db"][mask]
            box_data.append(vals)
            box_labels.append(f"{group}-{band}")
    ax2.boxplot(box_data, labels=box_labels, showfliers=False)
    ax2.set_ylabel("Power (dB)")
    ax2.set_title("Bandpower distributions (exclude bads)")
    ax2.tick_params(axis="x", rotation=45)

    plt.show()


if __name__ == "__main__":
    main()
