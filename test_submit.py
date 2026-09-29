import requests
import json
import time

base_url = "http://127.0.0.1:8000"
problem_id = ""

# 1. Generate problem
print("Generating problem...")
res = requests.post(f"{base_url}/api/coding/generate-problem", json={
    "title": "Two Sum",
    "statement": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
    "constraints": "2 <= nums.length <= 10^4",
    "sample": "Input: nums = [2,7,11,15], target = 9 \nOutput: 0 1"
})
print("Generate Problem Response:", res.status_code)
if res.status_code == 200:
    data = res.json()
    problem_id = data.get("problem_id")
    print("Problem ID:", problem_id)
else:
    print(res.text)
    exit(1)

test_cases = res.json().get("public_tests", [])
print("Public Tests:", test_cases)
if test_cases:
    expected_out = test_cases[0]["output"].strip()
    code = f"""
import sys
input_str = sys.stdin.read().strip()
print("{expected_out}")
"""
    
    print("\nSubmitting 'all' tests...")
    res2 = requests.post(f"{base_url}/api/coding/submit", json={
        "problem_id": problem_id,
        "language": "python",
        "code": code,
        "test_suite": "all"
    })
    print("Submit Response:", res2.status_code)
    print(json.dumps(res2.json(), indent=2))
    
    print("\nSubmitting 'samples' tests...")
    res3 = requests.post(f"{base_url}/api/coding/submit", json={
        "problem_id": problem_id,
        "language": "python",
        "code": code,
        "test_suite": "samples"
    })
    print("Submit Response (Samples):", res3.status_code)
    print(json.dumps(res3.json(), indent=2))
