"""The synthetic end-to-end smoke: train -> resume -> inference -> report (CPU, a few minutes).  Run with  pytest -m slow tests/test_smoke.py"""
import pytest

from inklingua.selftest import run_smoke


@pytest.mark.slow
def test_smoke(tmp_path):
    m = run_smoke(work_dir=str(tmp_path / "smoke"), keep=True)
    assert m["SYNTHETIC_SMOKE_TEST_NOT_FOR_PUBLICATION"] is True
