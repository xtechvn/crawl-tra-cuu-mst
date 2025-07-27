import requests
import json

# Base URL của API
BASE_URL = "http://localhost:8000"

def test_root():
    """Test root endpoint"""
    print("Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_search_api(tax_code="0123456789"):
    """Test search API"""
    print(f"\nTesting search API with tax code: {tax_code}")
    try:
        url = f"{BASE_URL}/search"
        data = {"tax_code": tax_code}
        
        response = requests.post(url, json=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Tax Crawler API Tests ===\n")
    
    # Test root endpoint
    root_ok = test_root()
    if not root_ok:
        print("❌ Root endpoint test failed. Make sure the API is running!")
        return
    
    print("✅ Root endpoint test passed!")
    
    # Test search API
    search_ok = test_search_api()
    if search_ok:
        print("✅ Search API test passed!")
    else:
        print("❌ Search API test failed!")
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    main() 