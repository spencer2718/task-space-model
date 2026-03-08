set more 1

/*

  Autor and Dorn, "The Growth of Low-Skill Service Jobs and the Polarization of the U.S. Labor Market"

  Employment Polarization Graphs for Figures 1A, 2A and 5A

  D. Autor, 3/22/2011

  Input data:
  emp-bypctile-1980-2005-rewt-0-czall.dta
  emp-bypctile-1980-2005-rewt-0-czlow-abovemean-RSHA.dta
  emp-bypctile-1980-2005-rewt-0-czhigh-abovemean-RSHA.dta
  emp-bypctile-1980-2005-rewt-1980-czall.dta
  emp-bypctile-1980-2005-rewt-1980-czlow-abovemean-RSHA.dta
  emp-bypctile-1980-2005-rewt-1980-czhigh-abovemean-RSHA.dta

  Output graphs:
  demp-pct-1980-2005-czall-color.gph
  demp-pct-1980-2005-rewt1980-czall-color.gph
  demp-pct-1980-2005-abovemean-RSHA-hilow-color.gph

*/

clear
set mem 5m
set scheme s2color

capture log close
log using ../log/plot-demp-bypctile-1980-2005.log, text replace


**************************************************************************************************
** Figures for all CZs
**************************************************************************************************

local subtit = ""
* Plot observed change and counterfactual change at 1980 service occupation employment
local bw = .75
local y1 = 1980
local y2 = 2005
local span1=-0.2
local span2=0.4
local int = 0.1
local tick=0.05
local rewtyr=1980

use ../dta/emp-bypctile-1980-2005-rewt-0-czall, clear
keep if year==`y1' | year==`y2'
gen byte rwt=0
append using ../dta/emp-bypctile-1980-2005-rewt-`rewtyr'-czall
keep if year==`y1' | year==`y2'
replace rwt=1 if rwt==.

* Calculate share changes for observed and reweighted distributions
sort rwt perc year
quietly by rwt perc: gen dsh = empshare-empshare[_n-1]
keep if year==`y2'
drop empshare
tab rwt, summ(dsh)
reshape wide dsh, i(perc) j(rwt)
label var perc "Occupation's Percentile in 1980 Wage Distribution"

* Get plotting points
lowess dsh0 perc, gen(pdsh0) bwidth(`bw') nograph
lowess dsh1 perc, gen(pdsh1) bwidth(`bw') nograph
label var pdsh0 "Observed change"
label var pdsh1 "Holding service emp at `rewtyr' level"

sort perc
tw scatter pdsh0 perc, connect(l) msymbol(o d) msize(small) ylabel(`span1'(`int')`span2') ymtick(`span1'(`tick')`span2') l1title("100 x Change in Employment Share", size(medsmall)) ytitle("") xtitle("Skill Percentile (Ranked by Occupational Mean Wage)", size(medsmall)) subtitle("`subtit'", size(medsmall)) title("Smoothed Changes in Employment by Skill Percentile `y1'-`y2'", size(medium)) saving(../gph/demp-pct-`y1'-`y2'-czall-color.gph,replace)
tw scatter pdsh0 pdsh1 perc, connect(l l) lpattern(solid dash) msymbol(sh i) msize(small small) ylabel(`span1'(`int')`span2') ymtick(`span1'(`tick')`span2') l1title("100 x Change in Employment Share", size(medsmall)) ytitle("") xtitle("Skill Percentile (Ranked by Occupational Mean Wage)", size(medsmall)) subtitle("`subtit'", size(medsmall)) title("Observed and Counterfactual Changes in Employment by Skill Percentile `y1'-`y2'", size(medium)) saving(../gph/demp-pct-`y1'-`y2'-rewt`rewtyr'-czall-color.gph,replace)
	
**************************************************************************************************
* Plot combined factuals for high and low RSH czones: 1980-2005
**************************************************************************************************

local suffix="-abovemean-RSHA"
use ../dta/emp-bypctile-1980-2005-rewt-0-czlow`suffix', clear
rename empshare emp_lo
sort year perc
merge year perc using ../dta/emp-bypctile-1980-2005-rewt-0-czhigh`suffix'
assert _merge==3
drop _merge
rename empshare emp_hi
keep if year==1980 | year==2005

* Calculate share changes for observed and reweighted distributions
sort perc year
quietly by perc: gen dsh_lo = emp_lo-emp_lo[_n-1]
quietly by perc: gen dsh_hi = emp_hi-emp_hi[_n-1]
label var perc "Occupation's Percentile in 1980 Wage Distribution"

* Get plotting points
lowess dsh_lo perc, gen(pdsh_lo) bwidth(`bw') nograph
lowess dsh_hi perc, gen(pdsh_hi) bwidth(`bw') nograph
label var pdsh_lo "Low Routine Share"
label var pdsh_hi "High Routine Share"
local subtit = "Commuting Zones Split on Mean Routine Share in 1980"

sort perc
tw scatter pdsh_lo pdsh_hi perc, connect(l l) lpattern(solid dash) msymbol(sh i) msize(small small) ylabel(-.2(.1).4) ymtick(-.2(.05).4) l1title("100 x Change in Employment Share", size(medsmall)) ytitle("") xtitle("Skill Percentile (Ranked by 1980 Occupational Mean Wage)", size(medsmall)) subtitle("`subtit'", size(medsmall)) title("Smoothed Changes in Employment by Skill Percentile 1980-2005", size(medium)) saving(../gph/demp-pct-1980-2005`suffix'-hilow-color.gph,replace)	

log close
