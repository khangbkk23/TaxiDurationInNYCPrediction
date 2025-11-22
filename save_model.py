import pickle
import os
def save_artifacts(model, scaler, feature_names):
   
    os.makedirs('artifacts', exist_ok=True)
    
    # Lưu model
    with open('artifacts/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Lưu scaler
    with open('artifacts/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Lưu feature names
    with open('artifacts/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("Đã lưu model thành công!")
    print(f"  - Model: artifacts/model.pkl")
    print(f"  - Scaler: artifacts/scaler.pkl")
    print(f"  - Features: artifacts/feature_names.pkl")

if __name__ == "__main__":
    save_artifacts(model, scaler, X_train.columns.tolist())
    pass