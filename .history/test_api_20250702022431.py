#!/usr/bin/env python3
"""
Test script to debug the Render deployment
Run this to test your API endpoints locally and on Render
"""

import requests
import json

def test_api(base_url):
    """Test the API endpoints"""
    print(f"\n🧪 Testing API at: {base_url}")
    
    # Test health endpoint
    try:
        print("\n1. Testing /health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False
    
    # Test debug endpoint
    try:
        print("\n2. Testing /debug endpoint...")
        response = requests.get(f"{base_url}/debug", timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ❌ Debug check failed: {e}")
    
    # Test prediction endpoint
    try:
        print("\n3. Testing /predict endpoint...")
        params = {
            'Age': 17,
            'Gender': 1,
            'Ethnicity': 0,
            'ParentalEducation': 2,
            'StudyTimeWeekly': 15.5,
            'Absences': 3,
            'Tutoring': 1,
            'ParentalSupport': 3,
            'Extracurricular': 1,
            'Sports': 1,
            'Music': 0,
            'Volunteering': 1
        }
        
        response = requests.get(f"{base_url}/predict", params=params, timeout=30)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Prediction successful!")
            print(f"   Grade: {result.get('predicted_grade')}")
            print(f"   Confidence: {result.get('confidence')}")
        else:
            print(f"   ❌ Prediction failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Prediction test failed: {e}")
    
    return True

if __name__ == "__main__":
    print("🚀 Student Performance Predictor API Test")
    
    # Test local development server
    print("\n" + "="*50)
    test_api("http://localhost:8080")
    
    # Test Render deployment
    print("\n" + "="*50)
    render_url = "https://student-performance-predictor-ai-live.onrender.com"
    test_api(render_url)
    
    print("\n🎯 Test completed!")
