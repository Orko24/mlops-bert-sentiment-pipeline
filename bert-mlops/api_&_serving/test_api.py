# test_api.py - Comprehensive API Testing
import requests
import json
import time
import concurrent.futures
import numpy as np
from datetime import datetime

API_URL = "http://localhost:8000"


def test_single_prediction():
    """Test single prediction endpoints"""
    test_cases = [
        {"text": "I absolutely love this product!", "expected": "positive"},
        {"text": "This is terrible, worst purchase ever", "expected": "negative"},
        {"text": "The product is okay, nothing special", "expected": "neutral"},
        {"text": "Amazing quality and fast shipping!", "expected": "positive"},
        {"text": "Complete waste of money", "expected": "negative"}
    ]

    print("🧪 Testing Single Predictions")
    print("-" * 30)

    for i, test_case in enumerate(test_cases, 1):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"text": test_case["text"], "return_confidence": True}
            )

            if response.status_code == 200:
                result = response.json()
                print(f"Test {i}: ✅")
                print(f"  Text: {test_case['text'][:50]}...")
                print(f"  Predicted: {result['sentiment']} (confidence: {result['confidence']:.3f})")
                print(f"  Latency: {result['prediction_time_ms']:.2f}ms")
                print()
            else:
                print(f"Test {i}: ❌ Status {response.status_code}")

        except Exception as e:
            print(f"Test {i}: ❌ Error: {str(e)}")


def test_load_performance(num_requests=100, max_workers=10):
    """Test API under load"""
    print(f"⚡ Load Testing ({num_requests} requests, {max_workers} concurrent)")
    print("-" * 50)

    test_texts = [
        "Great product, highly recommend!",
        "Not satisfied with the quality",
        "Average experience overall",
        "Excellent customer service",
        "Product arrived damaged"
    ]

    def make_request(text):
        start_time = time.time()
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"text": text},
                timeout=10
            )
            latency = (time.time() - start_time) * 1000
            return {
                'status_code': response.status_code,
                'latency_ms': latency,
                'success': response.status_code == 200
            }
        except Exception as e:
            return {
                'status_code': 0,
                'latency_ms': (time.time() - start_time) * 1000,
                'success': False,
                'error': str(e)
            }

    # Generate requests
    requests_data = [np.random.choice(test_texts) for _ in range(num_requests)]

    # Execute concurrent requests
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(make_request, requests_data))
    total_time = time.time() - start_time

    # Analyze results
    successful_requests = [r for r in results if r['success']]
    failed_requests = [r for r in results if not r['success']]

    if successful_requests:
        latencies = [r['latency_ms'] for r in successful_requests]

        print(f"📊 Load Test Results:")
        print(f"  Total Requests: {num_requests}")
        print(f"  Successful: {len(successful_requests)} ({len(successful_requests) / num_requests * 100:.1f}%)")
        print(f"  Failed: {len(failed_requests)}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Requests/sec: {num_requests / total_time:.2f}")
        print(f"  Average Latency: {np.mean(latencies):.2f}ms")
        print(f"  95th Percentile: {np.percentile(latencies, 95):.2f}ms")
        print(f"  99th Percentile: {np.percentile(latencies, 99):.2f}ms")
        print(f"  Max Latency: {max(latencies):.2f}ms")

    return results


def test_edge_cases():
    """Test edge cases and error handling"""
    print("🔬 Testing Edge Cases")
    print("-" * 25)

    edge_cases = [
        {"name": "Empty text", "data": {"text": ""}},
        {"name": "Very long text", "data": {"text": "a" * 1000}},
        {"name": "Special characters", "data": {"text": "!@#$%^&*()_+{}|:<>?[]\\;',./ ñáéíóú"}},
        {"name": "Numbers only", "data": {"text": "12345 67890"}},
        {"name": "Mixed languages", "data": {"text": "I love this producto es muy bueno"}},
        {"name": "HTML tags", "data": {"text": "<script>alert('test')</script> This is a test"}},
        {"name": "Multiple sentences", "data": {"text": "This is great. I love it. Highly recommend. Five stars!"}},
        {"name": "Emojis", "data": {"text": "I love this product! 😍🔥💯"}},
    ]

    for case in edge_cases:
        try:
            response = requests.post(f"{API_URL}/predict", json=case["data"])
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {case['name']}: {result['sentiment']} ({result['confidence']:.3f})")
            else:
                print(f"❌ {case['name']}: Status {response.status_code}")
        except Exception as e:
            print(f"❌ {case['name']}: Error {str(e)}")


def test_monitoring_endpoints():
    """Test monitoring and health endpoints"""
    print("📈 Testing Monitoring Endpoints")
    print("-" * 35)

    endpoints = [
        {"name": "Health Check", "url": f"{API_URL}/health", "method": "GET"},
        {"name": "Metrics", "url": f"{API_URL}/metrics", "method": "GET"},
        {"name": "Root", "url": f"{API_URL}/", "method": "GET"},
        {"name": "API Docs", "url": f"{API_URL}/docs", "method": "GET"},
    ]

    for endpoint in endpoints:
        try:
            if endpoint["method"] == "GET":
                response = requests.get(endpoint["url"])
            else:
                response = requests.post(endpoint["url"])

            if response.status_code == 200:
                print(f"✅ {endpoint['name']}: OK")
                if endpoint["name"] in ["Health Check", "Metrics", "Root"]:
                    try:
                        resp_json = response.json()
                        resp_text = str(resp_json)[:100]
                        print(f"   Response: {resp_text}...")
                    except:
                        print(f"   Response: {response.text[:100]}...")
            else:
                print(f"❌ {endpoint['name']}: Status {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint['name']}: Error {str(e)}")


def test_model_consistency():
    """Test model prediction consistency"""
    print("🔄 Testing Model Consistency")
    print("-" * 30)

    test_text = "I absolutely love this amazing product!"
    predictions = []

    print(f"Making 10 predictions for: '{test_text}'")

    for i in range(10):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"text": test_text}
            )

            if response.status_code == 200:
                result = response.json()
                predictions.append({
                    'sentiment': result['sentiment'],
                    'confidence': result['confidence'],
                    'latency': result['prediction_time_ms']
                })
        except Exception as e:
            print(f"Prediction {i + 1} failed: {e}")

    if predictions:
        sentiments = [p['sentiment'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        latencies = [p['latency'] for p in predictions]

        print(f"\n📊 Consistency Results:")
        print(f"  Unique Sentiments: {set(sentiments)}")
        print(f"  Confidence Range: {min(confidences):.3f} - {max(confidences):.3f}")
        print(f"  Confidence Std: {np.std(confidences):.4f}")
        print(f"  Average Latency: {np.mean(latencies):.2f}ms")
        print(f"  Latency Std: {np.std(latencies):.2f}ms")

        # Check consistency
        if len(set(sentiments)) == 1:
            print("✅ Predictions are consistent")
        else:
            print("⚠️ Predictions are inconsistent")


if __name__ == "__main__":
    print("🚀 Comprehensive API Testing Suite")
    print("=" * 40)
    print(f"Testing API at: {API_URL}")
    print(f"Timestamp: {datetime.now()}")
    print()

    # Run all tests
    test_single_prediction()
    print()

    test_edge_cases()
    print()

    test_monitoring_endpoints()
    print()

    test_model_consistency()
    print()

    # Load test (smaller for demo)
    test_load_performance(num_requests=50, max_workers=5)
    print()

    print("✅ All tests completed!")
    print("\n📊 To view detailed metrics, visit:")
    print(f"   API Metrics: {API_URL}/metrics")
    print(f"   API Docs: {API_URL}/docs")
    print(f"   Health Check: {API_URL}/health")