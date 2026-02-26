import os
import glob
import argparse

def count_subjects(npz_dir, dataset_type):
    # Lấy danh sách tất cả file .npz
    files = glob.glob(os.path.join(npz_dir, "*.npz"))
    filenames = [os.path.basename(f) for f in files]
    
    subjects = set()
    
    for name in filenames:
        if dataset_type.lower() == 'sleepedf':
            # Sleep-EDF: Lấy 5 hoặc 6 ký tự đầu (ví dụ: SC4001 từ SC4001E0)
            # Quy tắc: SC + 4 số đầu
            subj_id = name[:5] 
        elif dataset_type.lower() == 'hmc':
            # HMC: Tên file chính là ID bệnh nhân (ví dụ: SN001)
            subj_id = name.replace(".npz", "")
        else:
            # Mặc định lấy toàn bộ tên file làm ID
            subj_id = name.replace(".npz", "")
            
        subjects.add(subj_id)
    
    return len(filenames), len(subjects), sorted(list(subjects))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hmc_dir", help="Đường dẫn thư mục NPZ của HMC")
    ap.add_argument("--edf_dir", help="Đường dẫn thư mục NPZ của Sleep-EDF")
    args = ap.parse_args()

    print(f"{'Bộ dữ liệu':<15} | {'Số bản ghi':<15} | {'Số bệnh nhân':<15}")
    print("-" * 50)

    if args.hmc_dir:
        n_files, n_subjs, _ = count_subjects(args.hmc_dir, 'hmc')
        print(f"{'HMC':<15} | {n_files:<15} | {n_subjs:<15}")

    if args.edf_dir:
        n_files, n_subjs, _ = count_subjects(args.edf_dir, 'sleepedf')
        print(f"{'Sleep-EDF':<15} | {n_files:<15} | {n_subjs:<15}")

if __name__ == "__main__":
    main()