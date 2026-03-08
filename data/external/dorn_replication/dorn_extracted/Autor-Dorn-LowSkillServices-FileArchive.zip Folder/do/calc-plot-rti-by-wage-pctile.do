set more 1

version 9.2

* Set save variable
if !(_caller()<10) {
   local savemode = "old"
   disp "STATA FILE SAVING MODE: OLD FILE FORMAT"
}
capture log close
log using ../log/calc-plot-rti-by-wage-pctile.log,replace text

/*

  Autor and Dorn, "The Growth of Low-Skill Service Jobs and the Polarization of the U.S. Labor Market"

  Figure 4: Relationship between occupational skill and RTI

  D. Autor, 3/18/2011

  Input data:
  occ-means-by-decade-1980-2005-czall.dta

  Output data:
  RTI-bypctile-1980.dta

  Output graphs:
  RTI-byskill-1980-czall.gph

*/


clear
set mem 10m

********************************************************************************************************
** Prep matrix of occs ranked by mean wage indicating their employment shares at each percentile in 1980.
********************************************************************************************************

use ../dta/occ-means-by-decade-1980-2005-czall if year==1980, clear

* Cannot include occs with missing RTI
assert RTI!=.

* Assert that no farm occs are present
assert occ1990dd<473 | occ1990dd>498

* Assign weights
sort occ_mn_wg
summ occ_wt
egen yrwt=sum(occ_wt)
gen sharewt=occ_wt/yrwt
summ occ_wt sharewt

sort occ_mn_wg
gen runwt=sum(sharewt)
gen cumwt=runwt
replace cumwt=cumwt*100
summ cumwt

local occs = _N
gen left=cumwt[_n-1]
gen right=cumwt
replace left=0 in 1
gen margwt=right-left

quietly {
forvalues perc=1(1)100 {

  gen p`perc'=0
  forvalues occrank=1(1)`occs' {

    ** Four cases
    * 1. Right overlap
    if right[`occrank']>`perc'-1 & right[`occrank']<=`perc' & left[`occrank']<`perc'-1 {
      replace p`perc'= right-(`perc'-1) in `occrank'
    }

    * 2. Left overlap
    if left[`occrank']>=`perc'-1 & left[`occrank']<`perc' & right[`occrank']> `perc'{
      replace p`perc'= `perc'-left in `occrank'
    }

    * 3. Full overlap
    if left[`occrank'] <`perc'-1 & right[`occrank']>= `perc' {
      replace p`perc'= 1 in `occrank'
    }

    * 4. Full underlap (including exact match)
    if left[`occrank'] >= `perc'-1 & right[`occrank'] <= `perc' {
      replace p`perc'= right-left in `occrank'
    }
  }
}
}

/*
  The matrix we now need is psh by occ1990dd
  We multiply these shares by occupation shares of employment to get
  the total share in each percentile bin
  This share is 1 percent in 1980 by construction
  * Obtain total of shares by percentile
*/

forvalues perc=1(1)100 {
  gen psh`perc'=p`perc'/margwt
  egen empsh`perc'=sum(psh`perc'*margwt)
}

summ psh* empsh*
keep occ1990dd psh1-psh100 R33_3a hsd hsg smc clg year
sort occ1990dd

* Calculate weighted mean of R33_3a
forvalues perc=1(1)100 {
	capture drop percwt
	egen percwt=sum(psh`perc')
	* RTI measures that apply to all workers
	foreach v in R33_3a {
		egen perc_`v'_`perc' = sum(psh`perc'*`v'/percwt)
	}
}
keep if _n==1
drop psh* empsh* R* hsd hsg smc clg
desc , f
reshape long perc_R33_3a_, i(year) j(pctile)
rename perc_R33_3a_ perc_R33a

label var perc_R33a "Routine Occupation Share 1980"
sort pctile
label data "RSH by Percentile in 1980"
save ../dta/RTI-bypctile-1980, replace
desc
summ


********************************************************************************************************
** Make plots
********************************************************************************************************

set scheme s2color
lowess perc_R33a pctile, gen(pdr33) bwidth(.3) nograph
label var pdr33 "Routine Occupation Share"
tw scatter pdr33 pctile, connect(l) lpattern(solid) msymbol(sh) msymbol(d) msize(vsmall) title("Routine Task Intensity by Occupational Skill Percentile", size(medium)) xtitle("Skill Percentile (Ranked by 1980 Occupational Mean Wage)") ytitle("")  ylabel(0(0.10)0.7) l1title("Routine Occupation Share") saving(../gph/RTI-byskill-1980-czall.gph, replace)
* tw scatter pdr33 pctile, connect(l) lpattern(solid) msymbol(sh) msymbol(d) msize(vsmall) /* title("Routine Task Intensity by Occupational Skill Percentile", size(medium)) */ xtitle("Skill Percentile (Ranked by 1980 Occupational Mean Wage)") ytitle("")  ylabel(0(0.10)0.7) l1title("Routine Occupation Share") saving(../gph/RTI-byskill-1980-czall-notitle.gph, replace)
clear

log close
