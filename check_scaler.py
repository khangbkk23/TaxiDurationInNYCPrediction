import pickle
import os

def check_scaler():
    path = 'artifacts/scaler.pkl'
    
    if not os.path.exists(path):
        print("❌ Không tìm thấy file scaler.pkl")
        return

    with open(path, 'rb') as f:
        scaler = pickle.load(f)

    print("-" * 50)
    print("🕵️ SCALER ĐANG MONG ĐỢI CÁC CỘT SAU (Theo thứ tự):")
    print("-" * 50)
    
    # Kiểm tra thuộc tính lưu tên cột (có trong sklearn > 1.0)
    if hasattr(scaler, 'feature_names_in_'):
        expected_cols = scaler.feature_names_in_.tolist()
        print(expected_cols)
        
        print("\n👇 HÃY COPY LIST NÀY VÀO BIẾN 'SCALED_FEATURES' TRONG FILE main.py 👇")
        print("=" * 50)
        print(f"SCALED_FEATURES = {expected_cols}")
        print("=" * 50)
    else:
        print("⚠️ Scaler này được train bằng bản sklearn cũ hoặc input là numpy array nên không lưu tên cột.")
        print("Bạn phải nhớ chính xác thứ tự lúc train.")

if __name__ == "__main__":
    check_scaler()