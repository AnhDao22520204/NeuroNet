# import os, glob
# import mne

# RAW_DIR = r"D:\datasets\sleep-edfx-1.0.0\sleep-cassette"

# psg_files = sorted(glob.glob(os.path.join(RAW_DIR, "*-PSG.edf")))[:2]
# print("Found PSG:", len(glob.glob(os.path.join(RAW_DIR, '*-PSG.edf'))))

# for f in psg_files:
#     raw = mne.io.read_raw_edf(f, preload=False, verbose=False)
#     print("\n===", os.path.basename(f), "===")
#     print("sfreq:", raw.info["sfreq"])
#     print("channels:", raw.ch_names)

# -*- coding: utf-8 -*-
import os
import glob
import argparse
import mne
from collections import Counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="Thư mục chứa *-PSG.edf")
    ap.add_argument("--eeg", default="EEG Fpz-Cz")
    ap.add_argument("--eog", default="EOG horizontal")
    args = ap.parse_args()

    # 1. Tìm file
    psg_files = sorted(glob.glob(os.path.join(args.raw_dir, "*-PSG.edf")))
    hyp_files = sorted(glob.glob(os.path.join(args.raw_dir, "*-Hypnogram.edf")))

    print(f"\n{'='*20} SLEEP-EDF DATASET SURVEY {'='*20}")
    print(f"Số lượng file PSG phát hiện: {len(psg_files)}")
    print(f"Số lượng file Hypnogram phát hiện: {len(hyp_files)}")

    stats = {
        "sfreqs": [],
        "all_channels": Counter(),
        "missing_eeg": 0,
        "missing_eog": 0,
        "subjects": set()
    }

    for f in psg_files:
        subj_id = os.path.basename(f)[:6] # Ví dụ SC4001
        stats["subjects"].add(subj_id)
        
        try:
            raw = mne.io.read_raw_edf(f, preload=False, verbose=False)
            stats["sfreqs"].append(raw.info['sfreq'])
            stats["all_channels"].update(raw.ch_names)

            if args.eeg not in raw.ch_names: stats["missing_eeg"] += 1
            if args.eog not in raw.ch_names: stats["missing_eog"] += 1
        except:
            continue

    # --- BÁO CÁO THỐNG KÊ ---
    print(f"Tổng số bệnh nhân (subjects): {len(stats['subjects'])}")

    print(f"\n1. Tần số lấy mẫu (sfreq):")
    unique_fs = set(stats["sfreqs"])
    print(f"   - Các mức sfreq gốc: {unique_fs}")
    
    print(f"\n2. Tỷ lệ thiếu kênh mục tiêu:")
    print(f"   - Mục tiêu: EEG={args.eeg}, EOG={args.eog}")
    p_eeg = (stats['missing_eeg']/len(psg_files)*100)
    p_eog = (stats['missing_eog']/len(psg_files)*100)
    print(f"   - Subject thiếu EEG: {stats['missing_eeg']} ({p_eeg:.1f}%)")
    print(f"   - Subject thiếu EOG: {stats['missing_eog']} ({p_eog:.1f}%)")

    print(f"\n3. Các kênh xuất hiện nhiều nhất:")
    for ch, count in stats["all_channels"].most_common(7):
        print(f"   - {ch}: {count} files")

    if len(psg_files) != len(hyp_files):
        print(f"\n[WARNING] Số lượng PSG ({len(psg_files)}) và Hypnogram ({len(hyp_files)}) không khớp!")

if __name__ == "__main__":
    mne.set_log_level("WARNING")
    main()