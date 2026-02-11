from pathlib import Path

import matplotlib.pyplot as plt
import mne

#gdf_path = Path(r"E:\ISN 3A\PRDA\signals\steven.gdf")
gdf_path = Path(r"C:\Users\rayen\eeg\signals\clean_test_2_interface_working.gdf")
sensors_map_path = Path(r"C:\Users\rayen\eeg\sensors_coordinates.txt")
plot_signals = True
plot_duration_s = 10.0
plot_start_s = 0.0
plot_max_channels = 20
apply_plot_filter = True
plot_lfreq = 1.0
plot_hfreq = 40.0
plot_notch_freqs = [50, 100]


def main():
    raw = mne.io.read_raw_gdf(gdf_path, preload=False, verbose="error")
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
    eeg = raw.copy().pick("eeg")
    print(f"EEG channels: {len(eeg.ch_names)}")
    if eeg.ch_names:
        print("EEG channel names:", ", ".join(eeg.ch_names))
    if plot_signals and eeg.ch_names:
        raw_plot = raw.copy().load_data()
        if apply_plot_filter:
            raw_plot.filter(plot_lfreq, plot_hfreq, fir_design="firwin", verbose="error")
            raw_plot.notch_filter(plot_notch_freqs, verbose="error")
        raw_plot.pick("eeg")
        raw_plot.plot(
            duration=plot_duration_s,
            start=plot_start_s,
            n_channels=min(plot_max_channels, len(raw_plot.ch_names)),
            scalings="auto",
            show=False,
        )
        plt.show(block=True)


if __name__ == "__main__":
    main()
