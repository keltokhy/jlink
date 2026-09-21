# Offline Stata and R wrapper checks

Run from the repository root after `uv sync`:

```sh
uv run python tests/wrappers/run.py
uv run pytest tests/wrappers -q
```

The first command runs real `/usr/local/bin/Rscript` and `/usr/local/bin/stata-se`.
Use `--r` or `--stata` to run only one language. Both runtimes must be installed for
the default command. Pytest skips a runtime only when its executable is absent.

To keep diagnostics, supply a **new** directory (inside this owned test directory):

```sh
uv run python tests/wrappers/run.py --keep tests/wrappers/local-results
```

Do not commit generated diagnostics: Stata's startup banner includes license and
machine information. Ordinary runs use temporary directories that are removed.
The harness uses the current Python interpreter; `uv run` supplies pandas for
checking the Stata inputs. No new runtime dependencies are needed.

## What this proves

Each language runs twice: first with an executable named `jev-link` at the front
of PATH, then with only a fake `python3` that asserts it received `-m jlink`.
Neither route invokes the package's real CLI or accesses the network. The fake
writes one fixed link with IDs `00123` and `00007`, reads the wrapper's exported
input, and records arguments so shell quoting and temporary-file cleanup can be
checked. Its deliberate failure prints a `jlink:` error and exits with status 2.

Both also check that `style(rule)` / `style = "rule"` reaches the command without an entity, and
that a rule style with no definition, a missing entity and an unknown style stop before it.

R checks left data frames, both right data frames and file paths, numeric score
columns, text IDs, quoted definitions, CLI errors, empty results, unchanged input, and cleanup.
Stata checks the in-memory values, variable label and dataset characteristic
before/after success and failure, saving a links dataset, explicit merging with
string and numeric IDs, empty results, rejection of conflicting merge columns, and cleanup.
Paths include spaces and an apostrophe. Definitions include spaces, double quotes,
an apostrophe and shell metacharacters; a command-substitution sentinel must never
be created. R also tests literal dollar variables and backtick syntax. Stata users
must follow Stata's own macro quoting rules before values reach `jlink`.

Stata batch mode may return process status 0 even when a do-file fails. The
harness therefore requires an explicit completion marker and assertions in the
log, as well as a successful process status. It also checks that the deliberate
CLI error is present in the Stata log.

Verified on this machine with **R 4.5.1** and **StataNow/SE 19.5 (Apple Silicon)**.
This proves wrapper plumbing only. The real Linker, Jev API, accuracy of links,
Windows, and other runtime versions were not tested. The Stata wrapper currently
requires a POSIX shell (macOS or Unix); the R wrapper uses base R `system2` with
individually quoted arguments.
