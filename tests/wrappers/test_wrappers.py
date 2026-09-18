"""Native-language integration checks; skip only when that runtime is absent."""

import runpy
from pathlib import Path

import pytest

HARNESS = runpy.run_path(str(Path(__file__).with_name("run.py")))


@pytest.mark.parametrize("language,executable", [("r", "/usr/local/bin/Rscript"),
                                                 ("stata", "/usr/local/bin/stata-se")])
@pytest.mark.parametrize("fallback", [False, True], ids=["jev-link", "python-fallback"])
def test_native_wrapper(language, executable, fallback, tmp_path):
    if not Path(executable).exists():
        pytest.skip(f"{executable} is not installed")
    HARNESS["run"](language, fallback, tmp_path)
