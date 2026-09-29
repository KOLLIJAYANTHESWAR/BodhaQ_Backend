from types import SimpleNamespace

from app.services.problem_store import ProblemStore


session_a = "11111111-1111-1111-1111-111111111111"
session_b = "22222222-2222-2222-2222-222222222222"

problem = SimpleNamespace(
    public_tests=[],
    hidden_tests=[],
    statement="Isolation Test",
)

store = ProblemStore()

problem_id = store.save_problem(session_a, problem)

print("PROBLEM CREATED:", bool(problem_id))
print("A CAN READ:", store.get_problem(session_a, problem_id) is not None)
print("B CAN READ:", store.get_problem(session_b, problem_id) is not None)

blocked = False

try:
    store.append_tests(
        session_b,
        problem_id,
        [],
        [],
    )
except ValueError:
    blocked = True

print("B CANNOT APPEND:", blocked)
print("A STILL READS:", store.get_problem(session_a, problem_id) is not None)
print("DELETE BY B:", store.delete_problem(session_b, problem_id))
print("A AFTER B DELETE:", store.get_problem(session_a, problem_id) is not None)

# Cleanup
store.delete_problem(session_a, problem_id)