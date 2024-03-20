** This do-file is designed to implement an event study.

log using "${log}\3_DiD.smcl", replace
clear all
use "${wdata}\mu_MLPD.dta"

* The new minimum wage policy was implemented in July 2007, but MUS was done in May 2007.
gen post = 0
replace post = 1 if year >= 2008

* control variables
global xlist "age gender edu1 edu2 edu3 edu4 edu5 edu6 edu7 edu8  mar1 mar2 mar3  major1 major2 major3 major4 major5 major6 major7 major8 major9 major10  rel1 rel2 rel3 rel4 rel5 rel6 rel7 rel8 rel9 rel10 rel11 rel12 rel13 kid03 kid35 kid615 kid1517 kid18"
global plcae "county1 county2 county3 county4 county5 county6 county7 county8 county9 county10 county11 county12 county13 county14 county15 county16 county17 county18 county19"
global years "year2 year3 year4 year5 year6"



** Common Trend Test: significance of interaction term between year and treat
gen policy = year - 2008
rename hour Hour
rename employment Employment
rename earn Earning

keep if treat_GBC_t == 1 | treat_GBC_c == 1
replace treat = 1 if treat_GBC_t ==1
replace treat = 0 if treat_GBC_c ==1
drop if treat_GBC_t == 1 & treat_GBC_c == 1
gen DiD = treat*post


** Common Trend Test: mean comparison
* Test employment
egen mean_Employment=mean(Employment), by(year treat)
sum mean_Employment if year == 2007 & treat == 0
gen base_mean_Employment = r(mean)
sum mean_Employment if year == 2007 & treat == 1
replace base_mean_Employment = r(mean) if treat == 1
gen mean_Employment_sd = mean_Employment-base_mean_Employment

graph twoway (connect mean_Employment_sd year if treat==1,sort) (connect mean_Employment_sd year if treat==0,sort lpattern(dash)), ///
xline(2007,lpattern(dash) lcolor(gray)) ///
ytitle("Employment") xtitle("Year") ///
ylabel(,labsize(*0.75)) xlabel(,labsize(*0.75)) ///
legend(label(1 "Low-wage workers") label( 2 "Not-so-low-wage workers")) ///
xlabel(2005 (1) 2010)  graphregion(color(white)) ///
name(mean_cp_Employment, replace)
graph export "${pic}\mean_cp_Employment.png", replace

* Test LPR
egen mean_LPR=mean(LPR), by(year treat)
sum mean_LPR if year == 2007 & treat == 0
gen base_mean_LPR = r(mean)
sum mean_LPR if year == 2007 & treat == 1
replace base_mean_LPR = r(mean) if treat == 1
gen mean_LPR_sd = mean_LPR-base_mean_LPR

graph twoway (connect mean_LPR_sd year if treat==1,sort) (connect mean_LPR_sd year if treat==0,sort lpattern(dash)), ///
xline(2007,lpattern(dash) lcolor(gray)) ///
ytitle("Employment") xtitle("Year") ///
ylabel(,labsize(*0.75)) xlabel(,labsize(*0.75)) ///
legend(label(1 "Low-wage workers") label( 2 "Not-so-low-wage workers")) ///
xlabel(2005 (1) 2010)  graphregion(color(white)) ///
name(mean_cp_LPR, replace)
graph export "${pic}\mean_cp_LPR.png", replace

* Generate interaction term between year dummy and treat dummy
forvalues i = 3(-1)1{
  gen pre_`i' = (policy == -`i' & treat == 1) 
}

gen current = (policy == 0 & treat == 1)

forvalues j = 1(1)2{
  gen  post_`j' = (policy == `j' & treat == 1)
}

* Set pre_1(2007) as baseline year and regress
replace pre_1 = 0
* Test employment
reg Employment pre_* current post_* treat $years $plcae $xlist

* Visualization
coefplot, baselevels ///
keep(pre_* current post_*) ///
vertical ///
omitted ///
yline(0,lcolor(edkblue*0.8)) ///
xline(3, lwidth(vthin) lpattern(dash) lcolor(teal)) ///
ylabel(,labsize(*0.75)) xlabel(,labsize(*0.75)) ///
coeflabels(pre_3=2005 pre_2="2006" pre_1="2007" current= "2008" post_1="2009" post_2="2010") ///
ytitle("Policy Effect on Employment ({&theta})", size(medium)) ///
xtitle("Year", size(medium)) ///
addplot(line @b @at) /// add line between spots
ciopts(lpattern(dash) recast(rcap) msize(medium)) ///
graphregion(color(white)) ///
msymbol(circle_hollow)

graph rename pt_Employment
graph export "${pic}\parallel_trend_test_Employment.png", replace

* Test LPR
reg LPR pre_* current post_* treat $years $plcae $xlist

* Visualization
coefplot, baselevels ///
keep(pre_* current post_*) ///
vertical ///
omitted ///
yline(0,lcolor(edkblue*0.8)) ///
xline(3, lwidth(vthin) lpattern(dash) lcolor(teal)) ///
ylabel(,labsize(*0.75)) xlabel(,labsize(*0.75)) ///
coeflabels(pre_3=2005 pre_2="2006" pre_1="2007" current= "2008" post_1="2009" post_2="2010") ///
ytitle("Policy Effect on Employment ({&theta})", size(medium)) ///
xtitle("Year", size(medium)) ///
addplot(line @b @at) /// add line between spots
ciopts(lpattern(dash) recast(rcap) msize(medium)) ///
graphregion(color(white)) ///
msymbol(circle_hollow)

graph rename pt_LPR
graph export "${pic}\parallel_trend_test_LPR.png", replace


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

esttab Employment_0 Employment_1 Employment_2 LPR_0 LPR_1 LPR_2 using "${table}\reg.tex", se tex replace star(* .10 ** .05 *** .01)

log close