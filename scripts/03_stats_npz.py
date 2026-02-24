# scripts/03_stats_npz.py
import os
import glob
import argparse
import numpy as np
from collections import Counter

CLASS_NAMES = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
    print("Total subjects:", len(files))
    print("=" * 60)

    total_counter = Counter()

    for f in files:
        data = np.load(f, allow_pickle=True)
        y = data["y"]

        c = Counter(y.tolist())
        total_counter.update(c)

        print(os.path.basename(f))
        print("  epochs:", len(y))
        for k in sorted(c.keys()):
            name = CLASS_NAMES.get(int(k), str(k))
            print(f"    {name}: {c[k]}")
        print()

    print("=" * 60)
    print("TOTAL (all subjects)")
    print("Total epochs:", sum(total_counter.values()))
    for k in sorted(total_counter.keys()):
        name = CLASS_NAMES.get(int(k), str(k))
        print(f"{name}: {total_counter[k]}")

if __name__ == "__main__":
    main()
