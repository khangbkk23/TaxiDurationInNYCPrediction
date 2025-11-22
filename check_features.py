import pickle
import os

file_path = 'artifacts/features.pkl'

def inspect_features():
    if not os.path.exists(file_path):
        print(f"❌ Lỗi: Không tìm thấy file tại '{file_path}'")
        print("👉 Hãy kiểm tra lại xem bạn đã copy file vào thư mục artifacts chưa.")
        return

    try:
        with open(file_path, 'rb') as f:
            features = pickle.load(f)

        print(f"\nĐã load thành công! Tổng cộng có {len(features)} đặc trưng.")
        print("=" * 40)
        print(f"{'INDEX':<5} | {'FEATURE NAME'}")
        print("-" * 40)
        
        for i, name in enumerate(features):
            print(f"{i:<5} | {name}")
            
        print("=" * 40)
        
        # Kiểm tra nhanh các cột quan trọng
        important_cols = ['distance_km', 'pickup_hour', 'is_rush_hour']
        print("\nKiểm tra các cột quan trọng:")
        for col in important_cols:
            if col in features:
                print(f"Có cột '{col}' ở vị trí index {features.index(col)}")
            else:
                print(f"CẢNH BÁO: Thiếu cột '{col}' - Model sẽ dự đoán sai!")

    except Exception as e:
        print(f"❌ Lỗi khi đọc file: {e}")

if __name__ == "__main__":
    inspect_features()