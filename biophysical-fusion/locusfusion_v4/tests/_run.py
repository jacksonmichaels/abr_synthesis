"""Run a test module's test_* functions without pytest (abr_env has none).

    python -m locusfusion_v4.tests.test_model

Each test takes no arguments; shared setup is a functools.cache'd function in
the module. pytest, where installed, collects the same functions unchanged.
"""
import sys
import time
import traceback


def run(namespace, skip_reason=None):
    tests = [(k, v) for k, v in namespace.items() if k.startswith("test_") and callable(v)]
    if skip_reason:
        print(f"SKIPPED ({skip_reason}): {len(tests)} tests")
        sys.exit(0)
    fails = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"  ok    {name}  ({time.time() - t0:.1f}s)", flush=True)
        except Exception:
            fails += 1
            print(f"  FAIL  {name}", flush=True)
            traceback.print_exc()
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
