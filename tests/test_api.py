import os
import sys
import asyncio
from fastapi.testclient import TestClient

# Adjust python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.services.problem_store import problem_store

client = TestClient(app)

def run_tests():
    print("Running API Contract Tests...")

    # 1. Generate Problem
    print("\n--- Test Generate Problem ---")
    res = client.post("/api/coding/generate-problem", json={
        "title": "Two Sum",
        "statement": "",
        "constraints": "",
        "sample": ""
    })
    print(f"Status: {res.status_code}")
    
    problem_id = None
    if res.status_code == 200:
        data = res.json()
        problem_id = data.get('problem_id')
        print(f"Title: {data.get('title')}")
        print(f"Problem ID: {problem_id}")
        print(f"Public Tests: {len(data.get('public_tests', []))}")
        print(f"Hidden Test Count: {data.get('hidden_test_count')}")
        
        # Verify hidden tests are missing from response
        if 'hidden_tests' in data:
            print("ERROR: hidden_tests leaked in response!")
        else:
            print("Verified: hidden_tests missing from response (SECURE)")
            
        if 'hiddenTests' in data:
            print("ERROR: hiddenTests leaked in response!")
    else:
        print(res.text)
        return

    # 2. Check Backend Store
    print("\n--- Test Backend Store ---")
    stored = problem_store.get_problem(problem_id)
    if stored:
        print("Verified: Problem found in backend memory store.")
        print(f"Stored Hidden Tests count: {len(stored.hidden_tests)}")
    else:
        print("ERROR: Problem not found in store!")

    # 3. Analyze Code
    print("\n--- Test Analyze Code ---")
    res = client.post("/api/coding/analyze", json={
        "problem_statement": "Write a function that adds two numbers",
        "sample_test_case": "Input: a=1, b=2\nOutput: 3",
        "constraints": "none",
        "code": "def add(a, b): return a + b",
        "language": "python"
    })
    print(f"Status: {res.status_code}")
    if res.status_code == 200:
        print(f"Explanation Preview: {res.json().get('explanation')[:100]}...")
    else:
        print(res.text)

    # 4. Generate Testcases
    print("\n--- Test Generate Test Cases ---")
    res = client.post("/api/coding/testcases", json={
        "problem_id": problem_id,
        "problem_statement": "Write a function that adds two numbers",
        "sample_test_case": "Input: a=1, b=2\nOutput: 3",
        "constraints": "none",
        "code": "def add(a, b): return a + b",
        "language": "python"
    })
    print(f"Status: {res.status_code}")
    if res.status_code == 200:
        data = res.json()
        print(f"New Public Tests: {len(data.get('public_tests', []))}")
        print(f"Hidden Test Count returned: {data.get('hidden_test_count')}")
        if 'hidden_tests' in data:
            print("ERROR: hidden_tests leaked in testcases response!")
        else:
            print("Verified: hidden_tests missing from testcases response (SECURE)")
    else:
        print(res.text)

    # 5. Submit (Deterministic Judge)
    print("\n--- Test Submit ---")
    res = client.post("/api/coding/submit", json={
        "problem_id": problem_id,
        "language": "python",
        "code": "def main(): print('hello')"
    })
    print(f"Status: {res.status_code}")
    if res.status_code == 200:
        data = res.json()
        print(f"Submit Status: {data.get('status')}")
        print(f"Details: {data.get('details')}")
        if data.get('hidden_test_failed'):
            print("Verified: Returns hidden_test_failed=True safely.")
    else:
        print(res.text)


if __name__ == '__main__':
    run_tests()
