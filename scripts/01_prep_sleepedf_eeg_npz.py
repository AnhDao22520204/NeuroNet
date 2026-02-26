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
    "Sleep stage 4": 3,
    "Sleep stage R": 4
}

def find_hypnogram(psg_path, raw_dir):
    """Tìm file Hypnogram một cách linh hoạt hơn"""
    # Lấy tiền tố tên file, ví dụ: SC4001E0
    base_psg = os.path.basename(psg_path).replace("-PSG.edf", "")
    
    # Thử các trường hợp đặt tên phổ biến của Sleep-EDF
    candidates = [
        os.path.join(raw_dir, base_psg + "-Hypnogram.edf"), # Trường hợp tên giống hệt
        os.path.join(raw_dir, base_psg.replace("E0", "EC") + "-Hypnogram.edf"), # E0 -> EC
        os.path.join(raw_dir, base_psg.replace("E1", "EH") + "-Hypnogram.edf"), # E1 -> EH
    ]
    
    # Nếu vẫn không thấy, tìm bất cứ file nào bắt đầu bằng mã bệnh nhân (ví dụ SC4001)
    if base_psg.startswith("SC") or base_psg.startswith("ST"):
        subj_prefix = base_psg[:6] # SC4001
        pattern = os.path.join(raw_dir, f"{subj_prefix}*-Hypnogram.edf")
        candidates.extend(glob.glob(pattern))

    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def safe_bandpass(raw, l_freq=0.5, h_max=40.0):
    sfreq = float(raw.info["sfreq"])
    nyq = sfreq / 2.0
    h_freq = min(float(h_max), nyq - 0.5)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--eeg", default="EEG Fpz-Cz")
    ap.add_argument("--target_fs", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    psg_files = sorted(glob.glob(os.path.join(args.raw_dir, "*-PSG.edf")))
    saved, skipped = 0, 0

    for psg_path in psg_files:
        base = os.path.basename(psg_path).replace("-PSG.edf", "")
        hyp_path = find_hypnogram(psg_path, args.raw_dir)
        out_path = os.path.join(args.out_dir, f"{base}.npz")

        if not hyp_path:
            print(f"Skip {base}: Không tìm thấy file Hypnogram")
            skipped += 1; continue

        try:
            raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)
            if args.eeg not in raw.ch_names:
                print(f"Skip {base}: Thiếu kênh {args.eeg}. Kênh sẵn có: {raw.ch_names}")
                skipped += 1; continue
            
            raw.pick([args.eeg]).load_data()
            safe_bandpass(raw)
            if int(raw.info["sfreq"]) != args.target_fs:
                raw.resample(args.target_fs, verbose=False)

            ann = mne.read_annotations(hyp_path)
            raw.set_annotations(ann)
            
            # Lọc bỏ các nhãn không có trong bản đồ (như Sleep stage ?)
            event_id = {k: LABEL_MAP[k] + 1 for k in LABEL_MAP if k in ann.description}
            
            if not event_id:
                print(f"Skip {base}: Không tìm thấy nhãn hợp lệ trong {os.path.basename(hyp_path)}")
                skipped += 1; continue

            events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
            
            if len(events) == 0:
                print(f"Skip {base}: Không có sự kiện nào sau khi khớp nhãn")
                skipped += 1; continue

            epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0, 
                                tmax=30.0 - 1.0/args.target_fs, baseline=None, preload=True, verbose=False)
            
            x, y = epochs.get_data(), epochs.events[:, 2] - 1
            np.savez_compressed(out_path, x=x.astype(np.float32), y=y.astype(np.int64),
                                fs=np.array([args.target_fs]), ch_names=np.array([args.eeg]))
            print(f"Saved {base}: {x.shape}"); saved += 1
            
        except Exception as e:
            print(f"Error {base}: {e}"); skipped += 1

    print(f"\nDone! Saved: {saved}, Skipped: {skipped}")

if __name__ == "__main__":
    main()