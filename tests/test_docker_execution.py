import os
import sys
import subprocess
from fastapi.testclient import TestClient

# Adjust python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app

client = TestClient(app)

def run_exec(lang, code, stdin=""):
    return client.post("/api/coding/execute", json={
        "language": lang,
        "code": code,
        "stdin": stdin
    })

def verify():
    print("================== DOCKER EXECUTION VERIFICATION ==================")

    # 1. Java Execution & Input
    print("\n--- Testing Java Execution ---")
    java_code = """
import java.util.Scanner;
public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int a = sc.nextInt();
        int b = sc.nextInt();
        System.out.print(a + b);
    }
}
"""
    res = run_exec("java", java_code, "5 7")
    print(f"Java Result: {res.json().get('status')} | stdout: {res.json().get('stdout')}")

    # 2. Python Execution & Input
    print("\n--- Testing Python Execution ---")
    py_code = """
a, b = map(int, input().split())
print(f"Result: {a + b}")
"""
    res = run_exec("python", py_code, "10 20")
    print(f"Python Result: {res.json().get('status')} | stdout: {res.json().get('stdout').strip()}")

    # 3. Compilation Error
    print("\n--- Testing Java Compilation Error ---")
    bad_java = """
public class Main {
    public static void main(String[] args) {
        System.out.println(undefined_var);
    }
}
"""
    res = run_exec("java", bad_java)
    print(f"Status: {res.json().get('status')} | stderr snippet: {res.json().get('stderr')[:50]}")

    # 4. Runtime Error
    print("\n--- Testing Python Runtime Error ---")
    bad_py = "print(10 / 0)"
    res = run_exec("python", bad_py)
    print(f"Status: {res.json().get('status')} | stderr snippet: {res.json().get('stderr').strip().split(chr(10))[-1]}")

    # 5. Timeout
    print("\n--- Testing Timeout ---")
    timeout_py = "while True: pass"
    res = run_exec("python", timeout_py)
    print(f"Status: {res.json().get('status')} | stderr: {res.json().get('stderr')}")

    # 6. Network Isolation
    print("\n--- Testing Network Isolation ---")
    net_py = """
import urllib.request
try:
    urllib.request.urlopen("http://8.8.8.8", timeout=2)
    print("Network accessible!")
except Exception as e:
    print(f"Network error: {e}")
"""
    res = run_exec("python", net_py)
    print(f"Network Check stdout: {res.json().get('stdout').strip()}")

    # 7. Environment Variables
    print("\n--- Testing Environment Variables ---")
    env_py = "import os; print('GEMINI_API_KEY' in os.environ)"
    res = run_exec("python", env_py)
    print(f"GEMINI_API_KEY exposed: {res.json().get('stdout').strip()}")

    # 8. Output Limit
    print("\n--- Testing Output Limit ---")
    out_py = "print('A' * (150 * 1024))"
    res = run_exec("python", out_py)
    print(f"Status: {res.json().get('status')} | Output length: {len(res.json().get('stdout'))}")

    # 9. Container Cleanup
    print("\n--- Testing Container Cleanup ---")
    ps = subprocess.run(["docker", "ps", "-a", "--filter", "ancestor=python:3.10-alpine"], capture_output=True, text=True)
    containers = [line for line in ps.stdout.split('\n') if line.strip() and not line.startswith('CONTAINER')]
    print(f"Leftover Python containers: {len(containers)}")

if __name__ == '__main__':
    verify()
