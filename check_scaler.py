import pickle
import os
import numpy as np

def check_consistency():
    # Đường dẫn file
    model_path = 'artifacts/model.pkl'
    features_path = 'artifacts/features.pkl' # Hoặc feature_names.pkl tùy tên bạn lưu

    print("=" * 70)
    print("🕵️  KIỂM TRA KHỚP LỆNH (MODEL vs FEATURES FILE)")
    print("=" * 70)

    # 1. LOAD FEATURES.PKL
    list_from_file = []
    if os.path.exists(features_path):
        with open(features_path, 'rb') as f:
            list_from_file = pickle.load(f)
        print(f"\n📄 [features.pkl] chứa {len(list_from_file)} cột:")
        print(list_from_file)
    else:
        print(f"\n❌ Không tìm thấy file: {features_path}")

    # 2. LOAD MODEL.PKL
    list_from_model = []
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"\n🧠 [model.pkl] là loại: {type(model).__name__}")
        
        if hasattr(model, 'feature_names_in_'):
            list_from_model = model.feature_names_in_.tolist()
            print(f"👉 Model YÊU CẦU {len(list_from_model)} cột này (Bắt buộc đúng thứ tự):")
            print(list_from_model)
        else:
            print("⚠️ Model này không lưu tên cột bên trong (có thể do train bằng numpy array).")
    else:
        print(f"\n❌ Không tìm thấy file: {model_path}")

    # 3. SO SÁNH CHI TIẾT
    print("\n" + "=" * 70)
    print("⚖️  BẢNG SO SÁNH CHI TIẾT")
    print(f"{'INDEX':<5} | {'MODEL YÊU CẦU':<30} | {'TRONG FILE FEATURES.PKL':<30} | {'TRẠNG THÁI'}")
    print("-" * 70)

    # Lấy độ dài lớn nhất để loop
    max_len = max(len(list_from_model), len(list_from_file))

    all_match = True
    for i in range(max_len):
        m_col = list_from_model[i] if i < len(list_from_model) else "---"
        f_col = list_from_file[i] if i < len(list_from_file) else "---"
        
        status = "✅ OK"
        if m_col != f_col:
            status = "LỆCH"
            all_match = False
        
        print(f"{i:<5} | {m_col:<30} | {f_col:<30} | {status}")

    print("-" * 70)
    if all_match and max_len > 0:
        print("KẾT LUẬN: Tuyệt vời! Model và File khớp nhau 100%.")
    else:
        print("KẾT LUẬN: Có sự sai lệch! Hãy train lại và xuất file cùng lúc.")

if __name__ == "__main__":
    check_consistency()