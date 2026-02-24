import os, glob
import mne

RAW_DIR = r"D:\datasets\sleep-edfx-1.0.0\sleep-cassette"

psg_files = sorted(glob.glob(os.path.join(RAW_DIR, "*-PSG.edf")))[:2]
print("Found PSG:", len(glob.glob(os.path.join(RAW_DIR, '*-PSG.edf'))))

for f in psg_files:
    raw = mne.io.read_raw_edf(f, preload=False, verbose=False)
    print("\n===", os.path.basename(f), "===")
    print("sfreq:", raw.info["sfreq"])
    print("channels:", raw.ch_names)
