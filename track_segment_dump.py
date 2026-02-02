import re
from pathlib import Path

import mne
import numpy as np
import matplotlib.pyplot as plt

# --- config ---
gdf_path = Path(r"C:\Users\rayen\eeg\signals\clean_test_2_interface_working.gdf")
sensors_map_path = Path(r"C:\Users\rayen\eeg\sensors_coordinates.txt")
pre_start_s = 0.3
post_end_s = 0.3
print_full_data = False  # set False to avoid huge console output
print_units = "uV"  # "uV" or "V"
plot_psd = False
max_plots = None  # set an int to limit number of plotted segments
apply_average_ref = True
plot_spatial_overlay = True  # MNE evoked-style overlay with spatial colors (like quick_check.py)
apply_baseline = True  # baseline-correct using pre-marker window for overlay plot
apply_plot_filter = True  # 1-40 Hz band-pass + notch before plotting
plot_lfreq = 1.0
plot_hfreq = 40.0
plot_notch_freqs = [50, 100]  # change to [60, 120] if needed
apply_extra_lowpass = True  # extra smoothing for single-trial overlay
extra_lowpass_hz = 15.0  # lower = smoother, but can blur fast responses


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


def collect_track_markers(events, event_id):
    code_to_label = {code: label for label, code in event_id.items()}
    starts = {}
    ends = {}
    for sample, _, code in events:
        label = code_to_label.get(code, "")
        if not label.isdigit():
            continue
        value = int(label)
        if 34000 <= value < 35000:
            key = value % 1000
            starts.setdefault(key, []).append(sample)
        elif 35000 <= value < 36000:
            key = value % 1000
            ends.setdefault(key, []).append(sample)
    for key in starts:
        starts[key].sort()
    for key in ends:
        ends[key].sort()
    return starts, ends


def pair_markers(starts, ends):
    pairs = []
    for key in sorted(set(starts.keys()) | set(ends.keys())):
        s_list = starts.get(key, [])
        e_list = ends.get(key, [])
        si = 0
        ei = 0
        while si < len(s_list) and ei < len(e_list):
            if e_list[ei] < s_list[si]:
                ei += 1
                continue
            pairs.append((key, s_list[si], e_list[ei]))
            si += 1
            ei += 1
        if si < len(s_list) or ei < len(e_list):
            print(
                f"[warn] Unmatched markers for key {key:03d}: "
                f"starts_left={len(s_list) - si}, ends_left={len(e_list) - ei}"
            )
    return pairs


def main():
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose="error")
    apply_channel_mapping(raw, sensors_map_path)

    events, event_id = mne.events_from_annotations(raw, verbose="error")
    starts, ends = collect_track_markers(events, event_id)
    pairs = pair_markers(starts, ends)

    if not pairs:
        raise SystemExit("No start/end track markers found (34000-35999).")

    sfreq = raw.info["sfreq"]
    if print_full_data:
        np.set_printoptions(threshold=np.inf, linewidth=200)

    plotted = 0
    for key, start_sample, end_sample in pairs:
        start_code = 34000 + key
        end_code = 35000 + key
        start_adj = max(0, start_sample - int(pre_start_s * sfreq))
        end_adj = min(raw.n_times, end_sample + int(post_end_s * sfreq))
        pre_s = (start_sample - start_adj) / sfreq
        t_start = start_adj / sfreq
        t_end = end_adj / sfreq
        seg = raw.copy().crop(tmin=t_start, tmax=t_end)
        if apply_average_ref:
            seg.set_eeg_reference("average", projection=False, verbose="error")
        data = seg.get_data(picks="eeg")
        if print_units == "uV":
            data = data * 1e6
            unit_label = "uV"
        else:
            unit_label = "V"

        print(
            f"\nTrack key {key:03d} | start_code={start_code} end_code={end_code} "
            f"| t={t_start:.3f}s..{t_end:.3f}s | shape={data.shape} | units={unit_label}"
        )
        if print_full_data:
            print(data)

        if plot_psd:
            title = f"Track {key:03d} ({start_code}-{end_code})"
            seg_plot = seg.copy().load_data()
            if apply_plot_filter:
                seg_plot.filter(plot_lfreq, plot_hfreq, fir_design="firwin", verbose="error")
                seg_plot.notch_filter(plot_notch_freqs, verbose="error")
            seg_plot.compute_psd(fmax=60).plot(show=False)
            plt.show(block=True)
            plotted += 1
            if max_plots is not None and plotted >= max_plots:
                break

        if plot_spatial_overlay:
            title = f"Track {key:03d} ({start_code}-{end_code})"
            seg_plot = seg.copy().load_data()
            if apply_plot_filter:
                seg_plot.filter(plot_lfreq, plot_hfreq, fir_design="firwin", verbose="error")
                seg_plot.notch_filter(plot_notch_freqs, verbose="error")
            if apply_extra_lowpass:
                seg_plot.filter(None, extra_lowpass_hz, fir_design="firwin", verbose="error")
            seg_eeg = seg_plot.pick("eeg")
            evoked = mne.EvokedArray(
                seg_eeg.get_data(),
                seg_eeg.info,
                tmin=-pre_s,
                comment=title,
            )
            if apply_baseline and pre_s > 0:
                evoked.apply_baseline((None, 0.0))
            evoked.plot(
                spatial_colors=True,
                time_unit="s",
                show=False,
            )
            plt.show(block=True)
            plotted += 1
            if max_plots is not None and plotted >= max_plots:
                break


if __name__ == "__main__":
    main()
