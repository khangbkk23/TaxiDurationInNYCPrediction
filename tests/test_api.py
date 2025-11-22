"""
Test API đơn giản
"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    """Test prediction"""
    data = {
        "vendor_id": 2,
        "pickup_datetime": "2016-06-15 10:30:00",
        "passenger_count": 2,
        "pickup_longitude": -73.9776,
        "pickup_latitude": 40.7614,
        "dropoff_longitude": -73.9900,
        "dropoff_latitude": 40.7500
    }
    
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    result = response.json()
    assert result["success"] == True
    assert "duration_minutes" in result

if __name__ == "__main__":
    test_health()
    test_predict()
    print("✓ All tests passed!")