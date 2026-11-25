"""The fast invariant tests of the package (CPU, no download; ~30 s)."""
from inklingua.selftest import run_self_tests


def test_self_tests():
    run_self_tests(verbose=False)
