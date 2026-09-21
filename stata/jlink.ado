*! jlink 0.1.0 -- thin POSIX wrapper, requires Stata 14 or later
* Run on real Stata/SE 19.5 on macOS using an offline fake jev-link. Tested:
* temp files, quoted definitions, spaces in paths, saving/merge, unchanged data,
* empty links, command errors, python3 fallback, and style() with its entity()/define() checks.
* Also run once end to end against the real command and the live Jev API on examples/
* (2026-09-18); style(rule) has only been run against the fake. Not tested on Windows.
* Rerun: uv run python tests/wrappers/run.py --stata
program define jlink, rclass
    version 14
    syntax using/, ON(string) [ENTITY(string) DEFine(string) STYle(string) LEFTid(varname) ///
        RIGHTid(string) HOW(string) THReshold(real 0.5) BUDget(real 5) ///
        SAVing(string) REPLACE MERGE]
    if "`c(os)'" == "Windows" {
        display as error "jlink: this wrapper currently requires macOS or Unix"
        exit 198
    }
    if !inlist(`"`style'"', "", "identity", "rule") {
        display as error "jlink: style() must be identity or rule"
        exit 198
    }
    if `"`style'"' == "rule" & `"`define'"' == "" {
        display as error "jlink: style(rule) asks whether a pair satisfies define(), so define() is required"
        exit 198
    }
    if `"`style'"' != "rule" & `"`entity'"' == "" {
        display as error "jlink: entity() is required unless style(rule)"
        exit 198
    }
    if `"`how'"' == "" local how "one-to-one"
    if !inlist(`"`how'"', "one-to-one", "many-to-one", "one-to-many", "many-to-many") {
        display as error "jlink: how() must be one-to-one, many-to-one, one-to-many or many-to-many"
        exit 198
    }
    if missing(`threshold') | `threshold' < 0 | `threshold' > 1 | missing(`budget') | `budget' < 0 {
        display as error "jlink: threshold() must be between 0 and 1; budget() must be nonnegative"
        exit 198
    }
    if "`merge'" != "" & ("`leftid'" == "" | !inlist("`how'", "one-to-one", "many-to-one")) {
        display as error "jlink: merge needs leftid() and how(one-to-one) or how(many-to-one)"
        exit 198
    }
    if "`leftid'" != "" {
        isid `leftid'
        local idtype : type `leftid'
    }
    confirm file `"`using'"'
    tempfile left links errors status runner linkdata
    preserve
    quietly save `"`left'"', replace
    mata: _jlink_script()
    quietly shell /bin/sh "`runner'"
    capture confirm file `"`status'"'
    if _rc {
        display as error "jlink: could not run the command; check your shell and PATH"
        exit 198
    }
    tempname fh
    file open `fh' using `"`status'"', read text
    file read `fh' code
    file close `fh'
    if real("`code'") != 0 {
        capture noisily type `"`errors'"'
        display as error "jlink: command failed (exit `code')"
        exit 198
    }
    capture noisily type `"`errors'"'
    quietly import delimited using `"`links'"', clear varnames(1) stringcols(_all) encoding("utf-8")
    confirm variable left_id right_id p
    foreach column in p sim margin {
        capture confirm variable `column'
        if !_rc quietly destring `column', replace
    }
    local nlinks = _N
    if `"`saving'"' != "" quietly save `"`saving'"', `replace'
    if "`merge'" != "" {
        if substr("`idtype'", 1, 3) != "str" quietly destring left_id, replace
        if "`leftid'" != "left_id" rename left_id `leftid'
        quietly save `"`linkdata'"', replace
        unab linkvars : _all
    }
    restore
    if "`merge'" != "" {
        foreach column of local linkvars {
            if "`column'" != "`leftid'" confirm new variable `column'
        }
        preserve
        quietly merge 1:1 `leftid' using `"`linkdata'"', keep(master match) nogen
        restore, not
    }
    return scalar N_links = `nlinks'
    return local saving `"`saving'"'
    display as text "jlink: `nlinks' links; use saving() to keep the links or merge to attach them"
end

mata:
string scalar _jlink_quote(string scalar value)
{
    return("'" + subinstr(value, "'", "'" + char(34) + "'" + char(34) + "'") + "'")
}

void _jlink_script()
{
    real scalar fh, i
    string scalar cmd, input, output, cleanup
    string rowvector fields, options
    input = st_local("left") + ".dta"
    output = st_local("links") + ".csv"
    fh = fopen(st_local("runner"), "w")
    fput(fh, "#!/bin/sh")
    cleanup = "rm -f " + _jlink_quote(input) + " " + _jlink_quote(output)
    fput(fh, "trap " + _jlink_quote(cleanup) + " EXIT")
    fput(fh, "cp " + _jlink_quote(st_local("left")) + " " + _jlink_quote(input) + " || exit 1")
    fput(fh, "if command -v jev-link >/dev/null 2>&1; then set -- jev-link")
    fput(fh, "elif command -v python3 >/dev/null 2>&1; then set -- python3 -m jlink")
    fput(fh, "else printf '%s\n' 'jlink: install jev-link or python3 with jlink on PATH' > " +
        _jlink_quote(st_local("errors")))
    fput(fh, "printf '127\n' > " + _jlink_quote(st_local("status")) + "; exit 127; fi")
    cmd = char(34) + char(36) + "@" + char(34) + " link " + _jlink_quote(input) + " " +
        _jlink_quote(st_local("using")) + " -o " + _jlink_quote(output)
    fields = tokens(st_local("on"))
    for (i=1; i<=cols(fields); i++) cmd = cmd + " --on " + _jlink_quote(fields[i])
    options = ("entity", "define", "style", "how", "threshold", "budget", "leftid", "rightid")
    for (i=1; i<=cols(options); i++) {
        if (st_local(options[i]) != "") {
            if (options[i] == "leftid") cmd = cmd + " --left-id"
            else if (options[i] == "rightid") cmd = cmd + " --right-id"
            else cmd = cmd + " --" + options[i]
            cmd = cmd + " " + _jlink_quote(st_local(options[i]))
        }
    }
    cmd = cmd + " > " + _jlink_quote(st_local("errors")) + " 2>&1"
    fput(fh, cmd)
    fput(fh, "code=" + char(36) + "?")
    fput(fh, "if [ " + char(34) + char(36) + "code" + char(34) + " -eq 0 ]; then cp " +
        _jlink_quote(output) + " " + _jlink_quote(st_local("links")) + "; code=" + char(36) + "?; fi")
    fput(fh, "printf '%s\n' " + char(34) + char(36) + "code" + char(34) + " > " +
        _jlink_quote(st_local("status")))
    fclose(fh)
}
end
