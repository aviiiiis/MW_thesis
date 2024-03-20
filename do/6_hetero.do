** This do-file is designed for heterogeous analysis.

log using "${log}\12_hetero.smcl", replace
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
rename LPR LPR
rename employment Employment
rename earn Earning

keep if treat_GBC_t == 1 | treat_GBC_c == 1
replace treat = 1 if treat_GBC_t ==1
replace treat = 0 if treat_GBC_c ==1
drop if treat_GBC_t == 1 & treat_GBC_c == 1
gen DiD = treat*post


global xlist "age edu1 edu2 edu3 edu4 edu5 edu6 edu7 edu8  mar1 mar2 mar3  major1 major2 major3 major4 major5 major6 major7 major8 major9 major10  rel1 rel2 rel3 rel4 rel5 rel6 rel7 rel8 rel9 rel10 rel11 rel12 rel13 kid03 kid35 kid615 kid1517 kid18"

** DiD (repeated cross-sections): female
reg Employment DiD treat $years if gender==0
est sto Employment_0
reg Employment DiD treat $years $plcae  if gender==0
est sto Employment_1
reg Employment DiD treat $years $plcae $xlist   if gender==0
est sto Employment_2

reg LPR DiD treat $years if  gender==0
est sto LPR_0
reg LPR DiD treat $years $plcae  if  gender==0
est sto LPR_1
reg LPR DiD treat $years $plcae $xlist  if  gender==0 
est sto LPR_2		
	
		
esttab Employment_0 Employment_1 Employment_2 LPR_0 LPR_1 LPR_2  using "${table}\reg_female.tex", keep(DiD treat) se tex replace star(* .10 ** .05 *** .01)

** DiD (repeated cross-sections): male
reg Employment DiD treat $years  if gender==1
est sto Employment_0
reg Employment DiD treat $years $plcae    if gender==1
est sto Employment_1
reg Employment DiD treat $years $plcae $xlist   if gender==1
est sto Employment_2

reg LPR DiD treat $years  if  gender==1
est sto LPR_0
reg LPR DiD treat $years $plcae if  gender==1 
est sto LPR_1	
reg LPR DiD treat $years $plcae $xlist  if  gender==1 
est sto LPR_2			

global xlist "age gender mar1 mar2 mar3  major1 major2 major3 major4 major5 major6 major7 major8 major9 major10  rel1 rel2 rel3 rel4 rel5 rel6 rel7 rel8 rel9 rel10 rel11 rel12 rel13 kid03 kid35 kid615 kid1517 kid18"
		
esttab Employment_0 Employment_1 Employment_2 LPR_0 LPR_1 LPR_2  using "${table}\reg_male.tex", keep(DiD treat) se tex replace star(* .10 ** .05 *** .01)

** DiD (repeated cross-sections): high-educated
reg Employment DiD treat $years if eduyr >=9
est sto Employment_0
reg Employment DiD treat $years $plcae    if eduyr >=9
est sto Employment_1
reg Employment DiD treat $years $plcae $xlist   if eduyr >=9
est sto Employment_2

reg LPR DiD treat $years  if  eduyr >=9
est sto LPR_0
reg LPR DiD treat $years $plcae   if  eduyr >=9
est sto LPR_1
reg LPR DiD treat $years $plcae $xlist  if  eduyr >=9
est sto LPR_2		

		
esttab Employment_0 Employment_1 Employment_2 LPR_0 LPR_1 LPR_2  using "${table}\reg_highedu.tex", keep(DiD treat) se tex replace star(* .10 ** .05 *** .01)

** DiD (repeated cross-sections): low-educated
reg Employment DiD treat $years   if eduyr <9
est sto Employment_0
reg Employment DiD treat $years $plcae    if eduyr <9
est sto Employment_1
reg Employment DiD treat $years $plcae $xlist   if eduyr <9
est sto Employment_2

reg LPR DiD treat $years  if  eduyr <9
est sto LPR_0
reg LPR DiD treat $years $plcae   if  eduyr <9
est sto LPR_1
reg LPR DiD treat $years $plcae $xlist  if  eduyr <9
est sto LPR_2		

		
esttab Employment_0 Employment_1 Employment_2 LPR_0 LPR_1 LPR_2  using "${table}\reg_lowedu.tex", keep(DiD treat) se tex replace star(* .10 ** .05 *** .01)


global xlist "age gender edu1 edu2 edu3 edu4 edu5 edu6 edu7 edu8 major1 major2 major3 major4 major5 major6 major7 major8 major9 major10  rel1 rel2 rel3 rel4 rel5 rel6 rel7 rel8 rel9 rel10 rel11 rel12 rel13 kid03 kid35 kid615 kid1517 kid18"

** DiD (repeated cross-sections): not-married
reg Employment DiD treat $years  if marital==0
est sto Employment_0
reg Employment DiD treat $years $plcae    if marital==0
est sto Employment_1
reg Employment DiD treat $years $plcae $xlist   if marital==0
est sto Employment_2
reg LPR DiD treat $years if  marital==0
est sto LPR_0
reg LPR DiD treat $years $plcae   if  marital==0
est sto LPR_1
reg LPR DiD treat $years $plcae $xlist  if  marital==0
est sto LPR_2			
		
esttab Employment_0 Employment_1 Employment_2 LPR_0 LPR_1 LPR_2  using "${table}\reg_unmarried.tex", keep(DiD treat) se tex replace star(* .10 ** .05 *** .01)

** DiD (repeated cross-sections): married
reg Employment DiD treat $years   if marital==1
est sto Employment_0
reg Employment DiD treat $years $plcae    if marital==1
est sto Employment_1
reg Employment DiD treat $years $plcae $xlist   if marital==1
est sto Employment_2
reg LPR DiD treat $years if  marital==1
est sto LPR_0
reg LPR DiD treat $years $plcae   if  marital==1
est sto LPR_1
reg LPR DiD treat $years $plcae $xlist  if  marital==1
est sto LPR_2		
	
		
esttab Employment_0 Employment_1 Employment_2 LPR_0 LPR_1 LPR_2  using "${table}\reg_married.tex", keep(DiD treat) se tex replace star(* .10 ** .05 *** .01)

global xlist "gender edu1 edu2 edu3 edu4 edu5 edu6 edu7 edu8  mar1 mar2 mar3  major1 major2 major3 major4 major5 major6 major7 major8 major9 major10  rel1 rel2 rel3 rel4 rel5 rel6 rel7 rel8 rel9 rel10 rel11 rel12 rel13 kid03 kid35 kid615 kid1517 kid18"


** DiD (repeated cross-sections): youth
reg Employment DiD treat $years  if age<=35
est sto Employment_0
reg Employment DiD treat $years $plcae    if age<=35
est sto Employment_1
reg Employment DiD treat $years $plcae $xlist   if age<=35
est sto Employment_2

reg LPR DiD treat $years if  age<=35
est sto LPR_0
reg LPR DiD treat $years $plcae   if  age<=35
est sto LPR_1
reg LPR DiD treat $years $plcae $xlist  if  age<=35
est sto LPR_2		

		
esttab Employment_0 Employment_1 Employment_2 LPR_0 LPR_1 LPR_2  using "${table}\reg_35.tex", keep(DiD treat) se tex replace star(* .10 ** .05 *** .01)

** DiD (repeated cross-sections): middle-aged
reg Employment DiD treat $years if age>35 & age<=55
est sto Employment_0
reg Employment DiD treat $years $plcae    if age>35 & age<=55
est sto Employment_1
reg Employment DiD treat $years $plcae $xlist   if age>35 & age<=55
est sto Employment_2
reg LPR DiD treat $years if  age>35 & age<=55
est sto LPR_0
reg LPR DiD treat $years $plcae   if  age>35 & age<=55
est sto LPR_1
reg LPR DiD treat $years $plcae $xlist  if  age>35 & age<=55
est sto LPR_2		

		
esttab Employment_0 Employment_1 Employment_2 LPR_0 LPR_1 LPR_2  using "${table}\reg_35_55.tex", keep(DiD treat) se tex replace star(* .10 ** .05 *** .01)


** DiD (repeated cross-sections): older
reg Employment DiD treat $years  if age>55
est sto Employment_0
reg Employment DiD treat $years $plcae    if age>55
est sto Employment_1
reg Employment DiD treat $years $plcae $xlist   if age>55
est sto Employment_2
reg LPR DiD treat $years if  age>55
est sto LPR_0
reg LPR DiD treat $years $plcae   if  age>55
est sto LPR_1
reg LPR DiD treat $years $plcae $xlist  if  age>55
est sto LPR_2		
		
esttab Employment_0 Employment_1 Employment_2 LPR_0 LPR_1 LPR_2  using "${table}\reg_55.tex", keep(DiD treat) se tex replace star(* .10 ** .05 *** .01)
log close