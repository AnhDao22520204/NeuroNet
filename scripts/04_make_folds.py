import os
import glob
import json
import argparse
import numpy as np
from sklearn.model_selection import KFold, train_test_split

def get_subject_id(filename, dataset_type):
    """Trích xuất ID bệnh nhân từ tên file."""
    name = os.path.basename(filename).replace(".npz", "")
    if dataset_type.lower() == 'sleepedf':
        # Sleep-EDF: SC4001E0 -> ID là SC400 (5 ký tự đầu)
        return name[:5]
    else:
        # HMC hoặc các bộ khác: SN001 -> ID là SN001
        return name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--dataset", required=True, choices=['hmc', 'sleepedf'], help="Loại dataset để áp dụng quy tắc tách ID")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--val_size", type=float, default=0.2)
    args = ap.parse_args()

    # 1. Lấy danh sách tất cả các file .npz
    all_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(args.npz_dir, "*.npz"))])
    
    if not all_files:
        print("Lỗi: Không tìm thấy file .npz nào!")
        return

    # 2. Gom nhóm các file theo ID Bệnh nhân
    subj_to_files = {}
    for f in all_files:
        sid = get_subject_id(f, args.dataset)
        if sid not in subj_to_files:
            subj_to_files[sid] = []
        subj_to_files[sid].append(f)
    
    unique_subjects = sorted(list(subj_to_files.keys()))
    print(f"Tổng số bản ghi: {len(all_files)}")
    print(f"Tổng số bệnh nhân duy nhất: {len(unique_subjects)}")

    # 3. Chia K-Fold dựa trên danh sách Bệnh nhân duy nhất
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    splits = {}

    for fold_id, (train_val_idx, test_idx) in enumerate(kf.split(unique_subjects)):
        # Lấy ID bệnh nhân cho từng tập
        test_subj_ids = [unique_subjects[i] for i in test_idx]
        train_val_subj_ids = [unique_subjects[i] for i in train_val_idx]

        # Chia tập Train_Val thành Train và Val (dựa trên ID bệnh nhân)
        train_subj_ids, val_subj_ids = train_test_split(
            train_val_subj_ids, test_size=args.val_size, random_state=42, shuffle=True
        )

        # 4. Ánh xạ từ ID bệnh nhân quay lại danh sách các file tương ứng
        def map_ids_to_files(id_list):
            file_list = []
            for sid in id_list:
                file_list.extend(subj_to_files[sid])
            return sorted(file_list)

        splits[f"fold_{fold_id}"] = {
            "train": map_ids_to_files(train_subj_ids),
            "val": map_ids_to_files(val_subj_ids),
            "test": map_ids_to_files(test_subj_ids)
        }

        print(f"\nFold {fold_id}:")
        print(f"  Train: {len(train_subj_ids)} người ({len(splits[f'fold_{fold_id}']['train'])} file)")
        print(f"  Val  : {len(val_subj_ids)} người ({len(splits[f'fold_{fold_id}']['val'])} file)")
        print(f"  Test : {len(test_subj_ids)} người ({len(splits[f'fold_{fold_id}']['test'])} file)")

    # 5. Lưu JSON
    with open(args.out_file, "w") as f:
        json.dump(splits, f, indent=4)
    print(f"\nĐã lưu tại: {args.out_file}")

if __name__ == "__main__":
    main()