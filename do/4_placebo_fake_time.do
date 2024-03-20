** This do-file is designed to do placebo test.

log using "${log}\5_placebo_fake_time.smcl", replace
clear all
use "${wdata}\mu_MLPD.dta"

drop if year >= 2008

* set fake policy shock
gen post = 0
replace post = 1 if year >= 2007

* control variables
global xlist "age gender edu1 edu2 edu3 edu4 edu5 edu6 edu7 edu8  mar1 mar2 mar3  major1 major2 major3 major4 major5 major6 major7 major8 major9 major10  rel1 rel2 rel3 rel4 rel5 rel6 rel7 rel8 rel9 rel10 rel11 rel12 rel13 kid03 kid35 kid615 kid1517 kid18"
global plcae "county1 county2 county3 county4 county5 county6 county7 county8 county9 county10 county11 county12 county13 county14 county15 county16 county17 county18 county19"
global years "year2 year3 year4 year5 year6"

rename hour Hour
rename employment Employment
rename earn Earning

keep if treat_GBC_t == 1 | treat_GBC_c == 1
replace treat = 1 if treat_GBC_t ==1
replace treat = 0 if treat_GBC_c ==1
drop if treat_GBC_t == 1 & treat_GBC_c == 1
gen DiD = treat*post
save "${wdata}\r.dta", replace


use "${wdata}\r.dta", clear
** DiD (repeated cross-sections)
reg Employment DiD treat $years
est sto Employment_0
reg Employment DiD treat $years $plcae 
est sto Employment_1
reg Employment DiD treat $years $plcae $xlist 
est sto Employment_2

reg LPR DiD treat $years
est sto LPR_0
reg LPR DiD treat $years $plcae 
est sto LPR_1
reg LPR DiD treat $years $plcae $xlist 
est sto LPR_2	

esttab Employment_0 Employment_1 Employment_2 LPR_0 LPR_1 LPR_2 using "${table}\reg_rb_fake.tex", keep(DiD) se tex replace star(* .10 ** .05 *** .01)
erase "${wdata}\r.dta"


log close