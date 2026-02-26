import os
import glob
import argparse
import numpy as np
import mne
from collections import Counter

LABEL_MAP = {
    "Sleep stage W": 0, "Sleep stage N1": 1, "Sleep stage N2": 2,
    "Sleep stage N3": 3, "Sleep stage 3": 3, "Sleep stage 4": 3,
    "Sleep stage R": 4, "Sleep stage REM": 4
}
CLASS_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

def safe_bandpass(raw, l_freq=0.5, h_max=40.0):
    sfreq = float(raw.info["sfreq"])
    nyq = sfreq / 2.0
    h_freq = min(float(h_max), nyq - 0.5)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--eeg", default="EEG C4-M1")
    ap.add_argument("--target_fs", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    psg_files = sorted(glob.glob(os.path.join(args.raw_dir, "SN*.edf")))
    saved, skipped = 0, 0

    for psg_path in psg_files:
        if "sleepscoring" in psg_path: continue
        sid = os.path.basename(psg_path).replace(".edf", "")
        scor_path = os.path.join(args.raw_dir, f"{sid}_sleepscoring.edf")
        out_path = os.path.join(args.out_dir, f"{sid}.npz")

        if not os.path.exists(scor_path): continue

        try:
            raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)
            if args.eeg not in raw.ch_names:
                skipped += 1; continue
            
            raw.pick([args.eeg]).load_data()
            safe_bandpass(raw)
            if int(raw.info["sfreq"]) != args.target_fs:
                raw.resample(args.target_fs, verbose=False)

            ann = mne.read_annotations(scor_path)
            raw.set_annotations(ann)
            event_id = {k: LABEL_MAP[k] + 1 for k in LABEL_MAP if k in ann.description}
            events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)

            epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0, 
                                tmax=30.0 - 1.0/args.target_fs, baseline=None, preload=True, verbose=False)
            
            x, y = epochs.get_data(), epochs.events[:, 2] - 1
            np.savez_compressed(out_path, x=x.astype(np.float32), y=y.astype(np.int64),
                                fs=np.array([args.target_fs]), ch_names=np.array([args.eeg]))
            print(f"Saved {sid}: {x.shape}"); saved += 1
        except Exception as e:
            print(f"Error {sid}: {e}"); skipped += 1

    print(f"Done! Saved: {saved}, Skipped: {skipped}")

if __name__ == "__main__":
    main()