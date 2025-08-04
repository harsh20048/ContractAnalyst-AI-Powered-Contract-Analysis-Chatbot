#!/usr/bin/env python3
'''
Comprehensive test script for Mistral PDF Extraction Pipeline
'''

import requests
import time
import json

def test_server_health():
    '''Test if server is running.'''
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is healthy")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def test_mistral_status():
    '''Check Mistral model status.'''
    try:
        response = requests.get("http://localhost:8000/mistral/status")
        data = response.json()
        
        if data.get("loaded"):
            print("✅ Mistral 7B model is loaded")
            print(f"   📍 Model path: {data.get('model_path')}")
            return True
        else:
            print("⚠️  Mistral 7B model not loaded")
            print("   💡 Use: curl -X POST 'http://localhost:8000/mistral/load'")
            return False
    except Exception as e:
        print(f"❌ Error checking Mistral status: {e}")
        return False

def run_all_tests():
    '''Run comprehensive test suite.'''
    print("🧪 Running Mistral PDF Extraction Tests")
    print("=" * 50)
    
    tests = [
        ("Server Health", test_server_health),
        ("Mistral Status", test_mistral_status),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔬 {test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Your setup is ready!")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    run_all_tests()
