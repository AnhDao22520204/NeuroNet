# -*- coding: utf-8 -*-
import os
import glob
import argparse
import mne


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw_dir",
        required=True,
        help="Thư mục chứa các file HMC recordings (*.edf), ví dụ: D:\\DATASETS\\hmc_sleep\\recordings",
    )
    ap.add_argument(
        "--pattern",
        default="SN*.edf",
        help="Glob pattern để lọc file EDF (mặc định SN*.edf)",
    )
    ap.add_argument(
        "--max_files",
        type=int,
        default=3,
        help="Chỉ kiểm tra N file đầu tiên để test nhanh (mặc định 3). 0 = kiểm tra tất cả",
    )
    args = ap.parse_args()

    edf_files = sorted(glob.glob(os.path.join(args.raw_dir, args.pattern)))
    if not edf_files:
        print("Không tìm thấy EDF nào. Kiểm tra lại --raw_dir và --pattern.")
        return

    if args.max_files and args.max_files > 0:
        edf_files = edf_files[: args.max_files]

    print(f"Found EDF: {len(edf_files)} file(s)")
    print(f"raw_dir: {args.raw_dir}")
    print("-" * 80)

    for f in edf_files:
        base = os.path.basename(f)
        try:
            raw = mne.io.read_raw_edf(f, preload=False, verbose=False)
        except Exception as e:
            print(f"[ERROR] Cannot read: {base} -> {e}")
            print("-" * 80)
            continue

        sfreq = raw.info["sfreq"]
        chs = raw.ch_names

        print(f"=== {base} ===")
        print(f"sfreq: {sfreq}")
        print(f"n_channels: {len(chs)}")
        print("channels:")
        for ch in chs:
            print(f"  - {ch}")

        # gợi ý nhanh các kênh có thể là EEG/EOG (heuristic)
        eeg_like = [c for c in chs if "EEG" in c.upper()]
        eog_like = [c for c in chs if "EOG" in c.upper()]
        if eeg_like or eog_like:
            print("\n[Hint] EEG-like:", eeg_like[:10], ("..." if len(eeg_like) > 10 else ""))
            print("[Hint] EOG-like:", eog_like[:10], ("..." if len(eog_like) > 10 else ""))

        print("-" * 80)


if __name__ == "__main__":
    # giảm log rác của MNE
    mne.set_log_level("WARNING")
    main()
