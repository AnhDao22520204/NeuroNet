import os
import glob
import argparse
import numpy as np
from collections import Counter

# Định nghĩa các nhãn chuẩn
CLASS_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True, help="Thư mục chứa các file .npz đã tiền xử lý")
    args = ap.parse_args()

    # 1. Lấy danh sách file
    files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
    n_subjects = len(files)

    if n_subjects == 0:
        print(f"[Lỗi] Không tìm thấy file .npz nào tại: {args.npz_dir}")
        return

    # Khởi tạo các biến thống kê
    total_epochs = 0
    label_counter = Counter()
    epochs_per_subject = []
    sample_info = None

    print(f"Đang phân tích {n_subjects} subjects... Vui lòng đợi.")

    # 2. Quét từng file để thu thập dữ liệu
    for f in files:
        data = np.load(f, allow_pickle=True)
        y = data["y"]
        x = data["x"]
        fs = data["fs"][0]

        # Lưu thông tin mẫu từ file đầu tiên
        if sample_info is None:
            sample_info = {
                "shape": x.shape,
                "fs": fs,
                "ch_names": data["ch_names"]
            }

        n_epochs = len(y)
        total_epochs += n_epochs
        epochs_per_subject.append(n_epochs)
        label_counter.update(y.tolist())

    # 3. Tính toán các chỉ số
    avg_epochs = np.mean(epochs_per_subject)
    std_epochs = np.std(epochs_per_subject)
    
    # Xác định mức độ mất cân bằng (tìm lớp chiếm tỷ trọng lớn nhất)
    majority_class_idx = label_counter.most_common(1)[0][0]
    majority_class_name = CLASS_NAMES.get(majority_class_idx, str(majority_class_idx))
    imbalance_ratio = (label_counter[majority_class_idx] / total_epochs) * 100

    # 4. TRÌNH BÀY KẾT QUẢ BÁO CÁO
    print("\n" + "="*60)
    print(f"{'BÁO CÁO THỐNG KÊ DỮ LIỆU SAU TIỀN XỬ LÝ (NPZ)':^60}")
    print("="*60)

    print(f"1. THÔNG TIN CHUNG:")
    print(f"   - Tổng số Subjects (bệnh nhân): {n_subjects}")
    print(f"   - Tổng số Epochs (30s): {total_epochs:,}")
    print(f"   - Tần số lấy mẫu (fs) sau resample: {sample_info['fs']} Hz")
    print(f"   - Danh sách kênh đã trích xuất: {sample_info['ch_names']}")
    print(f"   - Cấu trúc dữ liệu mẫu (shape): {sample_info['shape']} (n_epochs, n_channels, T)")

    print(f"\n2. PHÂN BỐ NHÃN TOÀN BỘ DATASET:")
    print(f"   {'Giai đoạn':<12} | {'Số lượng (Count)':<15} | {'Tỷ lệ (%)':<10}")
    print(f"   {'-'*45}")
    
    # Kiểm tra xác nhận đúng 5 nhãn
    found_labels = sorted(label_counter.keys())
    for label_idx in range(5):
        name = CLASS_NAMES[label_idx]
        count = label_counter.get(label_idx, 0)
        percent = (count / total_epochs) * 100
        print(f"   {name:<12} | {count:<15,} | {percent:>8.2f}%")

    print(f"\n3. ĐẶC ĐIỂM ĐỊNH LƯỢNG:")
    print(f"   - Số epoch trung bình/subject: {avg_epochs:.1f} ± {std_epochs:.1f}")
    print(f"   - Lớp chiếm ưu thế (Majority): {majority_class_name} ({imbalance_ratio:.1f}%)")
    
    # Cảnh báo mất cân bằng
    if imbalance_ratio > 40:
        print(f"   - Cảnh báo: Mất cân bằng lớp cao ở giai đoạn {majority_class_name}.")
    
    print(f"   - Xác nhận 5 nhãn (0-4): {'ĐÚNG' if found_labels == [0,1,2,3,4] else 'SAI: ' + str(found_labels)}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()