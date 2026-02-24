# scripts/02_prep_hmc_eeg_eog_npz.py
# -*- coding: utf-8 -*-
import os
import glob
import argparse
import numpy as np
import mne
from collections import Counter

# HMC sleep staging thường dùng chuẩn: "Sleep stage W/N1/N2/N3/R"
LABEL_MAP = {
    "Sleep stage W": 0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3,
    "Sleep stage R": 4,   # REM
    "Sleep stage REM": 4, # phòng khi dataset ghi REM
    # fallback nếu có nơi ghi 1/2/3/4
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,   # gộp N4 -> N3
}

CLASS_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def safe_bandpass(raw: mne.io.BaseRaw, l_freq=1.0, h_max=50.0):
    """Filter an toàn theo Nyquist của sfreq hiện tại."""
    sfreq = float(raw.info["sfreq"])
    nyq = sfreq / 2.0
    h_freq = min(float(h_max), nyq - 0.5)  # luôn < Nyquist
    if h_freq <= l_freq:
        # nếu sfreq thấp quá (hiếm), bỏ highcut
        h_freq = None
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)


def find_pairs(raw_dir: str):
    """
    Tìm cặp:
      SNxxx.edf (PSG) + SNxxx_sleepscoring.edf (label)
    """
    psg_files = sorted(glob.glob(os.path.join(raw_dir, "SN*.edf")))
    pairs = []
    for psg in psg_files:
        base = os.path.basename(psg)
        # bỏ file scoring khỏi list PSG
        if base.endswith("_sleepscoring.edf"):
            continue
        stem = base.replace(".edf", "")  # SN001
        scor = os.path.join(raw_dir, f"{stem}_sleepscoring.edf")
        if os.path.exists(scor):
            pairs.append((psg, scor, stem))
    return pairs


def build_epochs_from_scoring(raw: mne.io.BaseRaw, ann: mne.Annotations, epoch_sec: int):
    """
    - raw đã pick đúng kênh và đã load_data()
    - ann lấy từ *_sleepscoring.edf
    """
    raw.set_annotations(ann)

    # chỉ lấy các label mà thực sự xuất hiện trong ann.description
    present = set(ann.description.tolist()) if hasattr(ann.description, "tolist") else set(ann.description)
    usable = [k for k in LABEL_MAP.keys() if k in present]

    if len(usable) == 0:
        return None, None, {"usable": [], "present": sorted(list(present))}

    # event_id phải >0
    event_id = {k: LABEL_MAP[k] + 1 for k in usable}
    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)

    if len(events) == 0:
        return None, None, {"usable": usable, "present": sorted(list(present))}

    # tạo epoch 30s
    tmax = float(epoch_sec) - 1.0 / raw.info["sfreq"]
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0.0,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    x = epochs.get_data()                 # (n_epochs, n_ch, T)
    y = epochs.events[:, 2].astype(int) - 1  # về 0..4
    return x, y, {"usable": usable, "present": sorted(list(present))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="folder chứa SNxxx.edf và SNxxx_sleepscoring.edf")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--eeg", default="EEG C4-M1")
    ap.add_argument("--eog", default="EOG E1-M2")
    ap.add_argument("--target_fs", type=int, default=100)
    ap.add_argument("--epoch_sec", type=int, default=30)
    ap.add_argument("--l_freq", type=float, default=1.0)
    ap.add_argument("--h_max", type=float, default=50.0)
    ap.add_argument("--max_subjects", type=int, default=0, help="0 = full, >0 = giới hạn để test")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--print_labels", action="store_true", help="in danh sách annotation label của từng subject")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pairs = find_pairs(args.raw_dir)
    if args.max_subjects and args.max_subjects > 0:
        pairs = pairs[:args.max_subjects]

    print("PSG to process:", len(pairs))
    print("raw_dir:", args.raw_dir)
    print("out_dir:", args.out_dir)
    print("EEG:", args.eeg, "| EOG:", args.eog, "| target_fs:", args.target_fs)

    saved, skipped = 0, 0
    total_counter = Counter()

    for psg_path, scor_path, sid in pairs:
        out_path = os.path.join(args.out_dir, f"{sid}.npz")
        if args.skip_existing and os.path.exists(out_path):
            continue

        print(f"\nProcessing: {sid}")
        print("  PSG :", os.path.basename(psg_path))
        print("  SCOR:", os.path.basename(scor_path))

        try:
            raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)
        except Exception as e:
            print("  Skip (cannot read PSG):", e)
            skipped += 1
            continue

        # check kênh
        need = [args.eeg, args.eog]
        missing = [ch for ch in need if ch not in raw.ch_names]
        if missing:
            print("  Skip (missing channels):", missing)
            print("  Available:", raw.ch_names)
            skipped += 1
            continue

        # pick trước để giảm RAM
        raw.pick(need)
        raw.load_data()  # cần load để filter/resample

        # filter + resample
        safe_bandpass(raw, l_freq=args.l_freq, h_max=args.h_max)
        if int(raw.info["sfreq"]) != args.target_fs:
            raw.resample(args.target_fs, verbose=False)

        # load annotations từ scoring
        try:
            ann = mne.read_annotations(scor_path)
        except Exception as e:
            print("  Skip (cannot read scoring annotations):", e)
            skipped += 1
            continue

        if args.print_labels:
            uniq = sorted(set(ann.description))
            print("  Labels in scoring:", uniq)

        # build epochs
        x, y, info = build_epochs_from_scoring(raw, ann, epoch_sec=args.epoch_sec)
        if x is None or y is None:
            print("  Skip (no usable events).")
            print("  Present labels:", info.get("present", [])[:30], "..." if len(info.get("present", [])) > 30 else "")
            skipped += 1
            continue

        # thống kê nhãn
        c = Counter(y.tolist())
        total_counter.update(c)
        show = {CLASS_NAMES.get(k, str(k)): v for k, v in sorted(c.items())}
        print("  Counts:", show)

        np.savez_compressed(
            out_path,
            x=x.astype(np.float32),              # (epochs, 2, T)
            y=y.astype(np.int64),                # (epochs,)
            fs=np.array([args.target_fs], dtype=np.int32),
            ch_names=np.array(need, dtype=object)
        )
        print(f"Saved: {out_path} x: {x.shape} y: {y.shape}")
        saved += 1

    print("\n" + "=" * 60)
    print("DONE")
    print("saved:", saved)
    print("skipped:", skipped)
    print("-" * 60)
    print("TOTAL (all saved subjects)")
    print("Total epochs:", sum(total_counter.values()))
    for k in range(5):
        print(f"{CLASS_NAMES[k]}: {total_counter.get(k, 0)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
