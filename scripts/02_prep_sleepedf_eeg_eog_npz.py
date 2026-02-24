# scripts/02_prep_sleepedf_eeg_eog_npz.py
# -*- coding: utf-8 -*-
import os
import glob
import argparse
import numpy as np
import mne

LABEL_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,   # N4 -> N3
    "Sleep stage R": 4,
}

def find_hypnogram(psg_path: str, raw_dir: str):
    stem = os.path.basename(psg_path).replace("-PSG.edf", "")  # SC4001E0
    candidates = []

    # 1) base-Hypnogram (fallback)
    candidates.append(os.path.join(raw_dir, f"{stem}-Hypnogram.edf"))

    # 2) sleep-cassette: E0->EC, E1->EH
    if stem.endswith("E0"):
        candidates.append(os.path.join(raw_dir, f"{stem[:-1]}C-Hypnogram.edf"))
    if stem.endswith("E1"):
        candidates.append(os.path.join(raw_dir, f"{stem[:-1]}H-Hypnogram.edf"))

    # 3) prefix fallback: SC4001*
    prefix = stem[:-2]
    candidates.extend(sorted(glob.glob(os.path.join(raw_dir, f"{prefix}*-Hypnogram.edf"))))

    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if os.path.exists(p):
            return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--eeg", default="EEG Fpz-Cz")
    ap.add_argument("--eog", default="EOG horizontal")
    ap.add_argument("--target_fs", type=int, default=100)
    ap.add_argument("--max_subjects", type=int, default=0)  # 0=full
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    psg_files = sorted(glob.glob(os.path.join(args.raw_dir, "*-PSG.edf")))
    if args.max_subjects and args.max_subjects > 0:
        psg_files = psg_files[:args.max_subjects]

    print("PSG to process:", len(psg_files))
    print("raw_dir:", args.raw_dir)
    print("out_dir:", args.out_dir)
    print("EEG:", args.eeg, "| EOG:", args.eog, "| target_fs:", args.target_fs)

    saved, skipped = 0, 0

    for psg_path in psg_files:
        base = os.path.basename(psg_path).replace("-PSG.edf", "")
        hyp_path = find_hypnogram(psg_path, args.raw_dir)
        out_path = os.path.join(args.out_dir, base + ".npz")

        if args.skip_existing and os.path.exists(out_path):
            print("Skip (exists):", base)
            skipped += 1
            continue

        if hyp_path is None:
            print("Skip (no hypnogram):", base)
            skipped += 1
            continue

        print("\nProcessing:", base)
        print("  PSG:", os.path.basename(psg_path))
        print("  HYP:", os.path.basename(hyp_path))

        try:
            # preload=False để nhẹ RAM
            raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)
            ann = mne.read_annotations(hyp_path)
            raw.set_annotations(ann)

            need = [args.eeg, args.eog]
            missing = [ch for ch in need if ch not in raw.ch_names]
            if missing:
                print("Skip (missing channels):", missing)
                # print("Available:", raw.ch_names)
                skipped += 1
                continue

            raw.pick(need)

            # filter an toàn theo Nyquist
            sfreq = float(raw.info["sfreq"])
            nyq = sfreq / 2.0
            h_freq = min(50.0, nyq - 0.5)
            if h_freq <= 1.0:
                print("Skip (bad nyquist): sfreq=", sfreq)
                skipped += 1
                continue
            raw.load_data()
            raw.filter(1.0, h_freq, verbose=False)

            if int(raw.info["sfreq"]) != args.target_fs:
                raw.resample(args.target_fs, verbose=False)

            event_id = {k: v + 1 for k, v in LABEL_MAP.items()}  # id>0
            events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)

            if len(events) == 0:
                print("Skip (no valid events)")
                skipped += 1
                continue

            epochs = mne.Epochs(
                raw, events, event_id=event_id,
                tmin=0.0, tmax=30.0 - 1.0 / raw.info["sfreq"],
                baseline=None, preload=True, verbose=False
            )

            x = epochs.get_data()       # (n_epochs, 2, T)
            y = epochs.events[:, 2] - 1 # 0..4

            np.savez_compressed(
                out_path,
                x=x.astype(np.float32),
                y=y.astype(np.int64),
                fs=np.array([args.target_fs], dtype=np.int32),
                ch_names=np.array(need, dtype=object),
            )
            print("Saved:", out_path, "x:", x.shape, "y:", y.shape)
            saved += 1

        except Exception as e:
            print("Skip (error):", base, "|", repr(e))
            skipped += 1
            continue

    print("\n" + "=" * 50)
    print("DONE")
    print("saved:", saved)
    print("skipped:", skipped)
    print("=" * 50)

if __name__ == "__main__":
    main()
