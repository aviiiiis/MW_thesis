** This Stata do-file is crafted to generate descriptive statistics tables.

log using "${log}\7_descriptive_table.smcl", replace

use "${wdata}\mu_98-10", clear
estpost summarize age gender eduyr marital employment LPR, detail
esttab using "${table}\mu_98-10.tex", cells("count mean(fmt(2)) sd(fmt(2)) p50(fmt(2)) min max") ///
		coeflabels(eduyr "education year" marital "marital status" earn "earning" LPR "labor force participation") ///
		title(MUS: 1998-2010\label{tab:mus9810}) nonumber noobs  replace

use "${wdata}\mu_ML", clear
estpost summarize age gender eduyr marital employment LPR, detail
esttab using "${table}\mu_ML.tex", cells("count mean(fmt(2)) sd(fmt(2)) p50(fmt(2)) min max") ///
		coeflabels(eduyr "education year" marital "marital status" earn "earning" LPR "labor force participation") ///
		title(Training (and validation) Dataset\label{tab:train}) nonumber noobs  replace

use "${wdata}\mu_ML_pre_del", clear
estpost summarize age gender eduyr marital employment LPR, detail
esttab using "${table}\mu_ML_pre_del.tex", cells("count mean(fmt(2)) sd(fmt(2)) p50(fmt(2)) min max") ///
		coeflabels(eduyr "education year" marital "marital status" earn "earning" LPR "labor force participation") ///
		title(Training (and validation) Dataset (Without Deletion)\label{tab:trainno}) nonumber noobs  replace


use "${wdata}\mu_PD", clear
drop if year<2005
estpost summarize age gender eduyr marital employment LPR, detail
esttab using "${table}\mu_PD.tex", cells("count mean(fmt(2)) sd(fmt(2)) p50(fmt(2)) min max") ///
		coeflabels(eduyr "education year" marital "marital status" earn "earning" LPR "labor force participation") ///
		title(Predicting Dataset\label{tab:prediction}) nonumber noobs  replace


log close