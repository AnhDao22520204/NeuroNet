# scripts/00_check_channels_isruc.py
# -*- coding: utf-8 -*-
import os, glob, argparse
import scipy.io

def guess_type(name: str):
    u = name.upper()
    if "ROC" in u or "LOC" in u or "EOG" in u:
        return "EOG-like"
    # các kênh EEG hay gặp trong ISRUC extracted channels
    if any(x in u for x in ["C3", "C4", "O1", "O2", "F3", "F4", "P3", "P4", "FZ", "CZ", "PZ", "OZ"]):
        return "EEG-like"
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--max_files", type=int, default=3)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.raw_dir, "*.mat")))
    if args.max_files > 0:
        files = files[:args.max_files]

    print(f"Found MAT: {len(files)} file(s)")
    print("raw_dir:", args.raw_dir)

    for fp in files:
        print("\n" + "=" * 80)
        print("File:", os.path.basename(fp))
        vars_ = scipy.io.whosmat(fp)

        eeg_cands, eog_cands = [], []
        for name, shape, dtype in vars_:
            t = guess_type(name)
            if t == "EEG-like":
                eeg_cands.append(name)
            elif t == "EOG-like":
                eog_cands.append(name)
            print(f"- {name:20s} shape={str(shape):12s} dtype={dtype:10s} -> {t}")

        print("\n[Hint] EEG candidates:", eeg_cands)
        print("[Hint] EOG candidates:", eog_cands)

if __name__ == "__main__":
    main()
