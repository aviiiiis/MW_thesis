** This Stata do-file is crafted to visualize the trends in minimum wage over multiple years.

clear all
set more off
cap log close
global sysdate = c(current_date)
graph set window fontface "Times New Roman"


log using "${log}\8_plot", replace


use "${rdata}\GDP.dta", clear

graph twoway (line gdp_growth year), ///
	xtitle("Year", size(medium)) xlabel(1970(5)2025) legend(off) graphregion(color(white)) ///
	xline(1997 2007, lpattern(dash))
graph export "${pic}\GDP_1.png", replace

graph twoway (line gdp_per year, yaxis(1)) (line gdp_growth year, yaxis(2)), ///
	xtitle("Year", size(medium)) xlabel(1970(5)2025) legend(on) graphregion(color(white)) ///
	xline(1997 2007, lpattern(dash))
graph export "${pic}\GDP_2.png", replace


use "${rdata}\minw.dta", clear

graph twoway (scatter minimumwagemonth year, msymbol(circle_hollow) mcolor(black)) ///
	(scatter minimumwagemonth year if year == 2007), ///
	ytitle("Minimum wage (monthly)", size(medium)) xtitle("Year", size(medium)) ///
	xlabel(1965(5)2025, labsize(*0.75)) ylabel(,labsize(*0.75)) ///
	legend(off) graphregion(color(white))
graph export "${pic}\minw_mon.png", replace


drop if year < 1992
graph twoway (scatter minimumwagehour year, msymbol(circle_hollow) mcolor(black)) ///
	(scatter minimumwagehour year if year == 2007), ///
	ytitle("Minimum wage (hourly)", size(medium)) xtitle("Year", size(medium)) ///
	xlabel(1990(5)2025, labsize(*0.75)) ylabel(,labsize(*0.75)) ///
	legend(off) graphregion(color(white))
graph export "${pic}\minw_hr.png", replace


estpost tabstat minimumwagemonth minimumwagehour, by(year)
esttab using "${table}\minw.tex", cells("minimumwagemonth minimumwagehour") noobs ///
nonumber varlabels(`e(labels)')  ///
drop(Total) varwidth(30) title(Minimum wage from 1992 to 2023 in Taiwan\label{tab:minw}) ///
 collab("Min. wage (month)" "Min. wage (hour)", lhs("year")) tex replace


log close