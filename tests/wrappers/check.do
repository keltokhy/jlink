version 14
set more off
about
args root right output
adopath ++ `"`root'/stata"'
clear
input str5 firm_id str20 name str10 city
"00123" "Acme & Sons" "Boston"
"00456" "Other firm" "New York"
end
label variable name "Firm name label survives"
char _dta[test] "Dataset characteristic survives"
generate int order = _n
quietly datasignature
local original `"`r(datasignature)'"'
local rule : environment JLINK_TEST_DEFINE
jlink using `"`right'"', on(name city=town) entity(firm) define(`"`rule'"') ///
    leftid(firm_id) rightid(record_id) threshold(.8) budget(0) saving(`"`output'"')
assert r(N_links) == 1
quietly datasignature
assert `"`r(datasignature)'"' == `"`original'"'
local name_label : variable label name
assert `"`name_label'"' == "Firm name label survives"
local characteristic : char _dta[test]
assert `"`characteristic'"' == "Dataset characteristic survives"
preserve
use `"`output'"', clear
assert _N == 1
assert left_id == "00123"
assert right_id == "00007"
assert p == .98
restore
jlink using `"`right'"', on(name city=town) style(rule) define(`"`rule'"') ///
    leftid(firm_id) rightid(record_id)
assert r(N_links) == 1
capture noisily jlink using `"`right'"', on(name city=town) style(rule) leftid(firm_id) rightid(record_id)
assert _rc == 198
capture noisily jlink using `"`right'"', on(name city=town) leftid(firm_id) rightid(record_id)
assert _rc == 198
capture noisily jlink using `"`right'"', on(name city=town) entity(firm) style(relation)
assert _rc == 198
quietly datasignature
assert `"`r(datasignature)'"' == `"`original'"'
local saved `"`c(pwd)'/run folder's files"'
jlink using `"`right'"', on(name city=town) entity(firm) leftid(firm_id) rightid(record_id) ///
    rundir(`"`saved'"')
assert r(N_links) == 1
confirm file `"`saved'/settings.json"'
capture noisily jlink using `"`right'"', on(name city=town) entity(firm) define("FAIL") ///
    leftid(firm_id) rightid(record_id)
assert _rc != 0
quietly datasignature
assert `"`r(datasignature)'"' == `"`original'"'
jlink using `"`right'"', on(name city=town) entity(firm) define("EMPTY") ///
    leftid(firm_id) rightid(record_id)
assert r(N_links) == 0
quietly datasignature
assert `"`r(datasignature)'"' == `"`original'"'
jlink using `"`right'"', on(name city=town) entity(firm) define(`"`rule'"') ///
    leftid(firm_id) rightid(record_id) merge
assert _N == 2
assert right_id == "00007" if firm_id == "00123"
assert missing(right_id) if firm_id == "00456"
assert p == .98 if firm_id == "00123"
assert order == _n
quietly datasignature
local merged `"`r(datasignature)'"'
capture noisily jlink using `"`right'"', on(name city=town) entity(firm) ///
    leftid(firm_id) rightid(record_id) merge
assert _rc != 0
quietly datasignature
assert `"`r(datasignature)'"' == `"`merged'"'
drop right_id block sim p source error margin
destring firm_id, replace
jlink using `"`right'"', on(name city=town) entity(firm) ///
    leftid(firm_id) rightid(record_id) how(many-to-one) merge
assert right_id == "00007" if firm_id == 123
assert _N == 2
display "STATA_WRAPPER_OK"
exit, clear
