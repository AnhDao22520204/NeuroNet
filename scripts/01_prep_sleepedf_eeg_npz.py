# -*- coding: utf-8 -*-
import os
import glob
import argparse
import numpy as np
import mne


# =============================
# Map về 5 lớp: W / N1 / N2 / N3 / REM
# =============================
LABEL_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # gộp N4 -> N3
    "Sleep stage R": 4,
}


# =============================
# Tìm đúng file hypnogram
# =============================
def find_hypnogram(psg_path, raw_dir):
    stem = os.path.basename(psg_path).replace("-PSG.edf", "")
    candidates = []

    # fallback trực tiếp
    candidates.append(os.path.join(raw_dir, f"{stem}-Hypnogram.edf"))

    # sleep-cassette chuẩn
    if stem.endswith("E0"):
        candidates.append(os.path.join(raw_dir, f"{stem[:-1]}C-Hypnogram.edf"))
    if stem.endswith("E1"):
        candidates.append(os.path.join(raw_dir, f"{stem[:-1]}H-Hypnogram.edf"))

    # fallback prefix
    prefix = stem[:-2]
    candidates.extend(sorted(glob.glob(os.path.join(raw_dir, f"{prefix}*-Hypnogram.edf"))))

    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# =============================
# MAIN
# =============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--eeg", default="EEG Fpz-Cz")
    parser.add_argument("--target_fs", type=int, default=100)
    parser.add_argument("--max_subjects", type=int, default=0)  # 0 = full
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    psg_files = sorted(glob.glob(os.path.join(args.raw_dir, "*-PSG.edf")))

    if args.max_subjects > 0:
        psg_files = psg_files[:args.max_subjects]

    print("========================================")
    print("PSG to process:", len(psg_files))
    print("raw_dir:", args.raw_dir)
    print("out_dir:", args.out_dir)
    print("EEG:", args.eeg)
    print("target_fs:", args.target_fs)
    print("========================================")

    saved, skipped = 0, 0

    for psg_path in psg_files:
        base = os.path.basename(psg_path).replace("-PSG.edf", "")
        hyp_path = find_hypnogram(psg_path, args.raw_dir)
        out_path = os.path.join(args.out_dir, base + ".npz")

        if hyp_path is None:
            print("Skip (no hypnogram):", base)
            skipped += 1
            continue

        print("\nProcessing:", base)

        try:
            # preload=False để tránh lỗi memory
            raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)

            # gắn annotation
            ann = mne.read_annotations(hyp_path)
            raw.set_annotations(ann)

            # pick 1 kênh EEG
            if args.eeg not in raw.ch_names:
                print("Skip (missing EEG):", args.eeg)
                skipped += 1
                continue

            raw.pick([args.eeg])

            # load data sau khi pick
            raw.load_data()

            # filter an toàn
            sfreq = float(raw.info["sfreq"])
            nyq = sfreq / 2.0
            h_freq = min(40.0, nyq - 0.5)  # an toàn
            raw.filter(0.5, h_freq, verbose=False)

            # resample
            if int(raw.info["sfreq"]) != args.target_fs:
                raw.resample(args.target_fs, verbose=False)

            # events
            event_id = {k: v + 1 for k, v in LABEL_MAP.items()}
            events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)

            if len(events) == 0:
                print("Skip (no valid events)")
                skipped += 1
                continue

            epochs = mne.Epochs(
                raw,
                events,
                event_id=event_id,
                tmin=0.0,
                tmax=30.0 - 1.0 / raw.info["sfreq"],
                baseline=None,
                preload=True,
                verbose=False,
            )

            x = epochs.get_data()          # (n_epochs, 1, T)
            y = epochs.events[:, 2] - 1    # về 0..4

            np.savez_compressed(
                out_path,
                x=x.astype(np.float32),
                y=y.astype(np.int64),
                fs=np.array([args.target_fs], dtype=np.int32),
                ch_names=np.array([args.eeg], dtype=object),
            )

            print("Saved:", base, "x:", x.shape)
            saved += 1

        except Exception as e:
            print("Skip (error):", base, "->", str(e))
            skipped += 1

    print("\n========================================")
    print("DONE")
    print("saved:", saved)
    print("skipped:", skipped)
    print("========================================")


if __name__ == "__main__":
    main()
