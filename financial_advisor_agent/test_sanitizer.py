"""Quick test to verify sanitize_for_llm works correctly."""
import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, '.')

from agent.utils import sanitize_for_llm, safe_json_dumps

# Test 1: normal text unchanged
normal = "Full market summary"
assert sanitize_for_llm(normal) == normal, "Normal text should pass through unchanged"
print("Test 1 PASSED: normal text unchanged")

# Test 2: surrogate in string is replaced
bad = "Full market summary\udcca extra"   # \udcca is a lone surrogate half
fixed = sanitize_for_llm(bad)
assert "\udcca" not in fixed, "Surrogate should be removed"
assert "Full market summary" in fixed, "Rest of text should remain"
print(f"Test 2 PASSED: surrogate replaced → '{fixed[:40]}'")

# Test 3: safe_json_dumps handles nested surrogates
data = {
    "headline": "RBI holds repo rate",
    "causal_factors": ["Rate\udcca hike", "Normal text"],
    "score": 0.72,
}
result = safe_json_dumps(data)
assert "\udcca" not in result
assert "RBI holds repo rate" in result
print(f"Test 3 PASSED: safe_json_dumps works → {result[:60]}...")

# Test 4: None input returns empty string
assert sanitize_for_llm(None) == ""
print("Test 4 PASSED: None → ''")

# Test 5: encode round-trip — should not throw
clean = sanitize_for_llm("Market summary")
clean.encode('utf-8')  # should NOT raise
print("Test 5 PASSED: clean string encodes to UTF-8 without error")

print("\n✅ ALL SANITIZER TESTS PASSED")
