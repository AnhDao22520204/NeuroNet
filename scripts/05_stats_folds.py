import os
import json
import argparse
import numpy as np
from collections import Counter

CLASS_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

def get_stats_for_list(subj_list, npz_dir):
    combined_counter = Counter()
    total_epochs = 0
    
    for sid in subj_list:
        path = os.path.join(npz_dir, f"{sid}.npz")
        if not os.path.exists(path):
            continue
        data = np.load(path)
        y = data["y"]
        combined_counter.update(y.tolist())
        total_epochs += len(y)
        
    return combined_counter, total_epochs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True, help="Thư mục chứa file .npz thực tế")
    ap.add_argument("--split_file", required=True, help="Đường dẫn file .json đã chia fold")
    args = ap.parse_args()

    with open(args.split_file, "r") as f:
        splits = json.load(f)

    for fold_id in sorted(splits.keys()):
        print(f"\n{'='*20} STATS FOR {fold_id.upper()} {'='*20}")
        
        for set_name in ["train", "val", "test"]:
            subj_list = splits[fold_id][set_name]
            counter, total = get_stats_for_list(subj_list, args.npz_dir)
            
            print(f"\n--- {set_name.upper()} ({len(subj_list)} subjects, {total} total epochs) ---")
            if total == 0:
                print("    No data found.")
                continue
                
            # In định dạng bảng cho dễ nhìn
            header = " | ".join([f"{CLASS_NAMES[i]:>5}" for i in range(5)])
            values = " | ".join([f"{counter.get(i, 0):>5}" for i in range(5)])
            percents = " | ".join([f"{(counter.get(i, 0)/total*100):>4.1f}%" for i in range(5)])
            
            print(f"    Stages: {header}")
            print(f"    Counts: {values}")
            print(f"    Ratio : {percents}")

if __name__ == "__main__":
    main()