# # -*- coding: utf-8 -*-
# import os
# import glob
# import argparse
# import mne


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument(
#         "--raw_dir",
#         required=True,
#         help="Thư mục chứa các file HMC recordings (*.edf), ví dụ: D:\\DATASETS\\hmc_sleep\\recordings",
#     )
#     ap.add_argument(
#         "--pattern",
#         default="SN*.edf",
#         help="Glob pattern để lọc file EDF (mặc định SN*.edf)",
#     )
#     ap.add_argument(
#         "--max_files",
#         type=int,
#         default=3,
#         help="Chỉ kiểm tra N file đầu tiên để test nhanh (mặc định 3). 0 = kiểm tra tất cả",
#     )
#     args = ap.parse_args()

#     edf_files = sorted(glob.glob(os.path.join(args.raw_dir, args.pattern)))
#     if not edf_files:
#         print("Không tìm thấy EDF nào. Kiểm tra lại --raw_dir và --pattern.")
#         return

#     if args.max_files and args.max_files > 0:
#         edf_files = edf_files[: args.max_files]

#     print(f"Found EDF: {len(edf_files)} file(s)")
#     print(f"raw_dir: {args.raw_dir}")
#     print("-" * 80)

#     for f in edf_files:
#         base = os.path.basename(f)
#         try:
#             raw = mne.io.read_raw_edf(f, preload=False, verbose=False)
#         except Exception as e:
#             print(f"[ERROR] Cannot read: {base} -> {e}")
#             print("-" * 80)
#             continue

#         sfreq = raw.info["sfreq"]
#         chs = raw.ch_names

#         print(f"=== {base} ===")
#         print(f"sfreq: {sfreq}")
#         print(f"n_channels: {len(chs)}")
#         print("channels:")
#         for ch in chs:
#             print(f"  - {ch}")

#         # gợi ý nhanh các kênh có thể là EEG/EOG (heuristic)
#         eeg_like = [c for c in chs if "EEG" in c.upper()]
#         eog_like = [c for c in chs if "EOG" in c.upper()]
#         if eeg_like or eog_like:
#             print("\n[Hint] EEG-like:", eeg_like[:10], ("..." if len(eeg_like) > 10 else ""))
#             print("[Hint] EOG-like:", eog_like[:10], ("..." if len(eog_like) > 10 else ""))

#         print("-" * 80)


# if __name__ == "__main__":
#     # giảm log rác của MNE
#     mne.set_log_level("WARNING")
#     main()

# -*- coding: utf-8 -*-
import os
import glob
import argparse
import mne
import numpy as np
from collections import Counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="Thư mục chứa file SNxxx.edf")
    ap.add_argument("--eeg", default="EEG C4-M1")
    ap.add_argument("--eog", default="EOG E1-M2")
    args = ap.parse_args()

    # 1. Tìm file
    all_edf = glob.glob(os.path.join(args.raw_dir, "*.edf"))
    psg_files = sorted([f for f in all_edf if "_sleepscoring" not in f])
    scoring_files = sorted([f for f in all_edf if "_sleepscoring" in f])

    print(f"\n{'='*20} HMC DATASET SURVEY {'='*20}")
    print(f"Tổng số file .edf hiện có: {len(all_edf)}")
    print(f"Số lượng file PSG phát hiện: {len(psg_files)}")
    print(f"Số lượng file Scoring (nhãn): {len(scoring_files)}")

    stats = {
        "sfreqs": [],
        "all_channels": Counter(),
        "missing_eeg": 0,
        "missing_eog": 0,
        "highpass": []
    }

    for f in psg_files:
        try:
            raw = mne.io.read_raw_edf(f, preload=False, verbose=False)
            stats["sfreqs"].append(raw.info['sfreq'])
            stats["all_channels"].update(raw.ch_names)
            stats["highpass"].append(raw.info['highpass'])

            if args.eeg not in raw.ch_names: stats["missing_eeg"] += 1
            if args.eog not in raw.ch_names: stats["missing_eog"] += 1
        except Exception as e:
            print(f"[Error] Không thể đọc file {os.path.basename(f)}: {e}")

    # --- BÁO CÁO THỐNG KÊ ---
    print(f"\n1. Tần số lấy mẫu (sfreq):")
    unique_fs = set(stats["sfreqs"])
    print(f"   - Các mức sfreq gốc: {unique_fs}")
    print(f"   - Trạng thái: {'Đồng nhất' if len(unique_fs)==1 else 'KHÔNG ĐỒNG NHẤT'}")

    print(f"\n2. Kiểm tra kênh mục tiêu (Target Channels):")
    print(f"   - Mục tiêu: EEG={args.eeg}, EOG={args.eog}")
    print(f"   - Thiếu EEG: {stats['missing_eeg']}/{len(psg_files)} ({(stats['missing_eeg']/len(psg_files)*100):.1f}%)")
    print(f"   - Thiếu EOG: {stats['missing_eog']}/{len(psg_files)} ({(stats['missing_eog']/len(psg_files)*100):.1f}%)")

    print(f"\n3. Danh sách tất cả kênh khả dụng (Top 10 phổ biến):")
    for ch, count in stats["all_channels"].most_common(10):
        print(f"   - {ch}: {count} files")

    print(f"\n4. Cảnh báo chất lượng:")
    unique_hp = set(stats["highpass"])
    if len(unique_hp) > 1:
        print(f"   - [WARNING] High-pass filter không đồng nhất giữa các file: {unique_hp}")
    else:
        print(f"   - High-pass filter ổn định tại mức: {unique_hp}")

if __name__ == "__main__":
    mne.set_log_level("WARNING")
    main()