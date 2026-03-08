*******************************************************************
* IPUMS Microdata Wage Regressions, 1980-2005
*******************************************************************

* David Dorn, version October 15, 2012
* Microdata-based wage regressions.


cap log close
set more off
clear
set memory 16g
set matsize 900

log using ../log/microwage1980_2005_regression.log, replace text

use ../dta/microwage1980_2005_data.dta, clear

* interaction of state and time dummy
gen tstate=statefip*t
xi i.tstate

* drop omitted categories of dummy variable systems
drop _Itstate_1
drop ed_0

* czone*occupation fixed effects (will absorb CZ-RSH*occupation interaction)
gen occ_czone=10*czone+(occ1_service==1)+2*(occ1_transconstr==1)+3*(occ1_mgmtproftech==1)+4*(occ1_clericretail==1)+5*(occ1_product==1)+6*(occ1_operator==1)

* occupation*time effects, CZ-RSH*occupation*time interactions
foreach c in occ1_service occ1_transconstr occ1_mgmtproftech occ1_clericretail occ1_product occ1_operator {
   gen t_`c'=t*`c'
   gen R33_`c'=txRA33_1950*`c'
}

***** TABLE 7, Panel B *****

areg ln_hrwage R33_occ* t_occ* txwkrs80min female marr race_nonwhite fborn ed_* exp* txfemale txmarr txrace_nonwhite txfborn txed_* txexp* _Its* [aw=timepwttot], absorb(occ_czone) cluster(czone)
areg ln_hrwage R33_occ* t_occ* txwkrs80min_m marr race_nonwhite fborn ed_* exp* txmarr txrace_nonwhite txfborn txed_* txexp* _Its* [aw=timepwttot] if female==0, absorb(occ_czone) cluster(czone)
areg ln_hrwage R33_occ* t_occ* txwkrs80min_f marr race_nonwhite fborn ed_* exp* txmarr txrace_nonwhite txfborn txed_* txexp* _Its* [aw=timepwttot] if female==1, absorb(occ_czone) cluster(czone)


log close
