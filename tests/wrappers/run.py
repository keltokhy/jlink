"""Run real Stata/R against a fake CLI, including a fake python3 fallback. No API calls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFINITION = 'Same firm: "Acme & Sons", O\'Brien; $HOME stays literal. $(touch INJECTION) `touch INJECTION`'


def shim(path: Path, fallback: bool) -> None:
    script = f"#!{sys.executable}\nimport runpy, sys\n"
    if fallback:
        script += "assert sys.argv[1:3] == ['-m', 'jlink'], sys.argv\ndel sys.argv[1:3]\n"
    script += f"runpy.run_path({str(HERE / 'fake_cli.py')!r}, run_name='__main__')\n"
    path.write_text(script)
    path.chmod(0o755)


def run(language: str, fallback: bool, work: Path) -> dict:
    mode = "python-fallback" if fallback else "jev-link"
    directory = work / f"{language} {mode}"
    directory.mkdir()
    binary = directory / "bin with spaces"
    binary.mkdir()
    shim(binary / ("python3" if fallback else "jev-link"), fallback)
    temporary = directory / "temporary files"
    temporary.mkdir()
    env = os.environ.copy()
    env.update(PATH=f"{binary}:/usr/bin:/bin", JLINK_TEST_DEFINE=(DEFINITION if language == "r" else
                   'Same firm: "Acme & Sons", O\'Brien; $(touch INJECTION)'),
               JLINK_TEST_LOG=str(directory / "invocations.jsonl"))
    # R itself refuses to start when R_TempDir contains spaces. Stata accepts it.
    if language == "stata":
        env.update(TMPDIR=str(temporary), STATATMP=str(temporary))
    right = directory / "company registry's data.dta"
    pd.DataFrame({"record_id": ["00007", "00008"], "name": ["Acme", "Other"],
                  "town": ["Boston", "New York"]}).to_stata(right, write_index=False)
    output = directory / "firm links.dta"
    start = time.perf_counter()
    if language == "r":
        argv = ["/usr/local/bin/Rscript", "--vanilla", str(HERE / "check.R"), str(ROOT / "r/jlink.R"),
                str(right)]
        result = subprocess.run(argv, cwd=directory, env=env, text=True, capture_output=True, timeout=60)
        text = result.stdout + result.stderr
        marker = "R_WRAPPER_OK"
    else:
        shutil.copyfile(HERE / "check.do", directory / "check.do")
        # Stata joins the arguments into its command language; compound double
        # quotes preserve file paths with spaces and apostrophes.
        quote = lambda value: '`"' + str(value) + '"\''
        argv = ["/usr/local/bin/stata-se", "-b", "do", "check.do", quote(ROOT), quote(right), quote(output)]
        result = subprocess.run(argv, cwd=directory, env=env, text=True, capture_output=True, timeout=90)
        log = directory / "check.log"
        text = (log.read_text(errors="replace") if log.exists() else "") + result.stdout + result.stderr
        marker = "STATA_WRAPPER_OK"
    (directory / "result.log").write_text(text)
    if result.returncode != 0 or marker not in text:
        raise AssertionError(f"{language} {mode} failed ({result.returncode}):\n{text}")
    if language == "stata":
        assert "jlink: registry.dta: column 'name' is missing" in text, "CLI diagnostic was hidden"
    assert not (directory / "INJECTION").exists(), "arguments were executed by the shell"
    invocations = [json.loads(line) for line in (directory / "invocations.jsonl").read_text().splitlines()]
    assert len(invocations) >= 3
    for record in invocations:
        assert not Path(record["left"]).exists(), f"temporary input leaked: {record}"
        assert not Path(record["output"]).exists(), f"temporary output leaked: {record}"
        if Path(record["right"]).resolve() != right.resolve():
            assert not Path(record["right"]).exists(), f"temporary right input leaked: {record}"
    assert not list(temporary.iterdir()), "native runtime temporary files leaked"
    seconds = time.perf_counter() - start
    print(f"PASS {language} {mode}: {len(invocations)} calls, {seconds:.2f}s", flush=True)
    return dict(language=language, mode=mode, calls=len(invocations), seconds=round(seconds, 3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r", action="store_true", help="run R only")
    parser.add_argument("--stata", action="store_true", help="run Stata only")
    parser.add_argument("--keep", type=Path,
                        help="retain logs in this new folder instead of temporary storage")
    args = parser.parse_args()
    languages = [name for name, flag in (("r", args.r), ("stata", args.stata))
                 if flag or not (args.r or args.stata)]
    if args.keep:
        args.keep.mkdir(parents=True, exist_ok=False)
        work = args.keep.resolve()
        results = [run(language, fallback, work) for language in languages for fallback in (False, True)]
        (work / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    else:
        with tempfile.TemporaryDirectory(prefix="jlink wrapper tests ") as temporary:
            work = Path(temporary)
            for language in languages:
                for fallback in (False, True):
                    run(language, fallback, work)


if __name__ == "__main__":
    main()
