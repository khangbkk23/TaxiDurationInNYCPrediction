import pickle
import os

def check_scaler():
    path = 'artifacts/scaler.pkl'
    
    if not os.path.exists(path):
        print("❌ Không tìm thấy file scaler.pkl")
        return

    with open(path, 'rb') as f:
        scaler = pickle.load(f)
    
    if hasattr(scaler, 'feature_names_in_'):
        expected_cols = scaler.feature_names_in_.tolist()
        print(expected_cols)

if __name__ == "__main__":
    check_scaler()