** This do-file is designed with the aim of cleaning and preparing data for analysis.

log using "${log}\1_clean.smcl", replace
use "${rdata}\mu1991to2020.dta", clear

* Min Wage changed in 1997 and 2007, use data in 1998-2010 to implement ML
drop if year < 1998 | year > 2010

* variables with no observation
drop kid611 kid1214

* drop workers with positive(0) wage but 0(positive) hour worked
drop if (hour == 0 & earn > 0) | (earn == 0 & hour > 0)

* female:0, male:1
gen gender = 0 if sex == 2
replace gender = 1 if sex == 1

* employment:1 Unemployment:0
gen employment = 0
replace employment = 1 if earn != 0

* LPR
gen LPR = 0
replace LPR = 1 if employment == 1 | work == 5

* drop workers work more than 168 hours
drop if hour > 168

* single:1, divorce:3, death:4, married:2
gen marital = 0 if mar == 1 | mar == 3 | mar == 4
replace marital = 1 if mar == 2

* taipei：63
gen taipei = 0
replace taipei = 1 if county == 63

* low-wage:<1.2*17280, middle-wage:1.2*17280~2*17280, high-wage:>2*17280
gen treat = 0
replace treat = 1 if earn >= 17280*1.2 & earn < 17280*2
replace treat = 2 if earn >= 17280*2

save "${wdata}\mu_98-10", replace

* edu dummy
tabulate edu, generate(edu)

* marital dummy
tabulate mar, generate(mar)

* county dummy
tabulate county, generate(county)

* major dummy
tab major, generate(major)

* character dummy
tabulate rel, generate(rel)


keep year age gender edu* eduyr taipei marital kid* work* rel* mar* county* major* stat1 earn hour treat employment LPR
label drop _all

preserve

* use data in 1998-2004 to train
keep if year < 2005
save "${wdata}\mu_ML_pre_del", replace
* keep if workers aged 15-65
keep if age <= 65 & age >= 15
* workers in private and government sector
keep if stat1 == 3 | stat1 == 4
* drop workers with no wage, deficit and Maximum
drop if earn  == 0 | earn  == 1 | earn == 999999
save "${wdata}\mu_ML", replace
restore

* use data in 2005-2010 to predict
* keep if workers aged 15-65
keep if age <= 65 & age >= 15
keep if year >= 2005
drop if earn == 999999
* year dummy 
tabulate year, generate(year)
save "${wdata}\mu_PD", replace

log close