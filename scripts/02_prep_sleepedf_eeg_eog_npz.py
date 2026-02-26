import os
import glob
import argparse
import numpy as np
import mne

# Thống nhất 5 lớp nhãn
LABEL_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,   # N4 -> N3
    "Sleep stage R": 4,
}

def find_hypnogram(psg_path, raw_dir):
    """Logic tìm Hypnogram thông minh xử lý lỗi lệch tên file của Sleep-EDF"""
    base_psg = os.path.basename(psg_path).replace("-PSG.edf", "")
    
    # Danh sách các ứng viên có thể là file nhãn
    candidates = [
        os.path.join(raw_dir, base_psg + "-Hypnogram.edf"),
        os.path.join(raw_dir, base_psg.replace("E0", "EC") + "-Hypnogram.edf"), # Quy tắc E0 -> EC
        os.path.join(raw_dir, base_psg.replace("E1", "EH") + "-Hypnogram.edf"), # Quy tắc E1 -> EH
    ]
    
    # Tìm kiếm theo tiền tố nếu các quy tắc trên thất bại
    subj_prefix = base_psg[:6] # Lấy mã bệnh nhân, ví dụ SC4001
    pattern = os.path.join(raw_dir, f"{subj_prefix}*-Hypnogram.edf")
    candidates.extend(glob.glob(pattern))

    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def safe_bandpass(raw, l_freq=0.5, h_max=40.0):
    """Filter an toàn đảm bảo không vi phạm định luật Nyquist"""
    sfreq = float(raw.info["sfreq"])
    nyq = sfreq / 2.0
    h_freq = min(float(h_max), nyq - 0.5)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="Thư mục chứa file EDF gốc")
    ap.add_argument("--out_dir", required=True, help="Thư mục lưu file NPZ")
    ap.add_argument("--eeg", default="EEG Fpz-Cz")
    ap.add_argument("--eog", default="EOG horizontal")
    ap.add_argument("--target_fs", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    psg_files = sorted(glob.glob(os.path.join(args.raw_dir, "*-PSG.edf")))
    
    print(f"Bắt đầu xử lý {len(psg_files)} file PSG...")
    saved, skipped = 0, 0

    for psg_path in psg_files:
        base = os.path.basename(psg_path).replace("-PSG.edf", "")
        hyp_path = find_hypnogram(psg_path, args.raw_dir)
        out_path = os.path.join(args.out_dir, f"{base}.npz")

        if not hyp_path:
            print(f"  [SKIP] {base}: Không tìm thấy file Hypnogram phù hợp.")
            skipped += 1; continue

        try:
            # 1. Đọc metadata
            raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)
            
            # 2. Kiểm tra đủ 2 kênh
            need = [args.eeg, args.eog]
            missing = [ch for ch in need if ch not in raw.ch_names]
            if missing:
                print(f"  [SKIP] {base}: Thiếu kênh {missing}. Kênh hiện có: {raw.ch_names[:5]}...")
                skipped += 1; continue
            
            # 3. Trích xuất, Lọc và Resample
            raw.pick(need).load_data()
            safe_bandpass(raw)
            if int(raw.info["sfreq"]) != args.target_fs:
                raw.resample(args.target_fs, verbose=False)

            # 4. Xử lý nhãn
            ann = mne.read_annotations(hyp_path)
            raw.set_annotations(ann)
            
            # Chỉ lấy các sự kiện thuộc 5 lớp quan tâm
            event_id = {k: LABEL_MAP[k] + 1 for k in LABEL_MAP if k in ann.description}
            if not event_id:
                print(f"  [SKIP] {base}: Không tìm thấy giai đoạn ngủ hợp lệ (W/N1/N2/N3/R).")
                skipped += 1; continue

            events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
            
            # 5. Cắt Epoch 30s
            epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0, 
                                tmax=30.0 - 1.0/args.target_fs, baseline=None, preload=True, verbose=False)
            
            x = epochs.get_data()       # Kích thước: (n_epochs, 2, 3000)
            y = epochs.events[:, 2] - 1 # Chuyển nhãn về 0-4
            
            # 6. Lưu file nén
            np.savez_compressed(
                out_path, 
                x=x.astype(np.float32), 
                y=y.astype(np.int64),
                fs=np.array([args.target_fs], dtype=np.int32), 
                ch_names=np.array(need, dtype=object)
            )
            print(f"  [OK] {base}: Đã lưu {x.shape} vào {os.path.basename(out_path)}")
            saved += 1
            
        except Exception as e:
            print(f"  [ERROR] {base}: {repr(e)}")
            skipped += 1

    print(f"\n{'='*30}")
    print(f"HOÀN THÀNH!")
    print(f"Thành công: {saved}")
    print(f"Bỏ qua    : {skipped}")
    print(f"{'='*30}")

if __name__ == "__main__":
    main()