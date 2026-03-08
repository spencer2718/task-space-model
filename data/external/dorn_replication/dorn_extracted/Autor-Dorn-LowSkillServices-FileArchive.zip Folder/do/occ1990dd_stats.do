*******************************************************************
*
* Autor and Dorn, "The Growth of Low-Skill Service Jobs and the Polarization of the U.S. Labor Market"
*
* Occupation statistics for Tables 1-2, Appendix Tables 1-2, Appendix Table 4 (Panel B), and Figure 3
*
* David Dorn, May 20, 2011
*
*******************************************************************

set memory 200m

log using ../log/occ1990dd_stats.log, replace text


use ../dta/occ1990dd_data2012.dta, clear


***** TABLE 1 - Employment shares and wages by occupation group *****

foreach v in occ1_mgmtproftech occ1_product occ1_transconstr occ1_operator occ1_clericretail occ1_service {
   foreach y in 1950 1970 1980 1990 2000 2005 {
      disp "*** employment share `v' in `y' (percentage points) ***"
      quietly egen sh_empl_`v'_`y'=sum(sh_empl`y'*`v'*100)
      tab sh_empl_`v'_`y'
      quietly drop sh_empl_`v'_`y'
   }
}

foreach v in occ1_mgmtproftech occ1_product occ1_transconstr occ1_operator occ1_clericretail occ1_service {
   foreach y in 1950 1970 1980 1990 2000 2005 {
      disp "*** average log wage `v' in `y' ***"
      summ avg_hrwage`y' if `v'==1 [aw=sh_empl`y']
   }
}


***** TABLE 2 - Tasks by occupation group *****

* compute mean and std.dev. in 1980
foreach v in RTIa task_abstract task_routine task_manual {
   summ `v' [aw=sh_empl1980]
   gen mean_`v'=r(mean)
   gen sd_`v'=r(sd)
   gen `v'_std=(`v' - mean_`v')/sd_`v'
}

foreach v in occ1_mgmtproftech occ1_product occ1_transconstr occ1_operator occ1_clericretail occ1_service {
   foreach k in RTIa_std task_abstract_std task_routine_std task_manual_std {
      disp "*** task `k' in `v' ***"
      summ `k' if `v'==1  [aw=sh_emplnc1980]
   }
}


***** APPENDIX TABLE 1 - Employment shares and wages by occupation group, non-college workers ******

foreach v in occ1_mgmtproftech occ1_product occ1_transconstr occ1_operator occ1_clericretail occ1_service {
   foreach y in 1950 1970 1980 1990 2000 2005 {
      disp "*** employment share among non-college `v' in `y' (percentage points) ***"
      quietly egen sh_emplnc_`v'_`y'=sum(sh_emplnc`y'*`v'*100)
      tab sh_emplnc_`v'_`y'
      quietly drop sh_emplnc_`v'_`y'
   }
}

foreach v in occ1_mgmtproftech occ1_product occ1_transconstr occ1_operator occ1_clericretail occ1_service {
   foreach y in 1950 1970 1980 1990 2000 2005 {
      disp "*** average log wage among non-college `v' in `y' ***"
      summ avg_hrwage`y'_nc if `v'==1 [aw=sh_emplnc`y']
   }
}


***** APPENDIX TABLE 2 - Occuptations with highest RTI score / Low-skill and high-skill occs with lowest RTI score *****

* high RTI
gen counter=1
* w/o small, farm, supervisory, residual occs
replace RTIa=. if (occ1990dd>=473 & occ1990dd<=498) | (sh_empl1980<1/660)
replace RTIa=. if occ1990dd==22 | occ1990dd==89 | occ1990dd==169 | occ1990dd==159 | occ1990dd==243 | occ1990dd==349 | occ1990dd==503 | occ1990dd==558 | occ1990dd==628
gen RTIa_large=RTIa
gsort -RTIa_large
gen sum_counter=sum(counter)
forvalues v=1(1)10 {
   disp "rank `v'"
   tab occ1990dd if sum_counter==`v'
}
drop sum_counter

* low-skill and high-skill low RTI
summ sh_edunc1980 [aw=sh_empl1980] if (occ1990dd<473 | occ1990dd>498)
gen meanedunc=r(mean)
gen RTIa_nc_large=RTIa if sh_edunc1980>meanedunc
gen RTIa_c_large=RTIa if sh_edunc1980<=meanedunc

gsort +RTIa_nc_large
gen sum_counter=sum(counter)
forvalues v=1(1)10 {
   disp "rank `v'"
   tab occ1990dd if sum_counter==`v'
}
drop sum_counter

gsort +RTIa_c_large
gen sum_counter=sum(counter)
forvalues v=1(1)10 {
   disp "rank `v'"
   tab occ1990dd if sum_counter==`v'
}
drop sum_counter


***** APPENDIX TABLE 4 (Panel B) - Employment shares of detailed service occupations, non-college workers *****

foreach var of varlist occ3* {
   foreach y in 1980 2005 {
      disp "*** employment share non-college `var' in `y' (percentage points) ***"
      quietly egen sh_emplnc_`y'=sum(sh_emplnc`y'*`var'*100)
      tab sh_emplnc_`y'
      quietly drop sh_emplnc_`y'
   }
}


***** FIGURE 3 - Employment change by decade in bottom quintile of the occupational distribution *****
*** note: this file generates data points only; see file plot-svc-growth-barchart-p20.do for graph ***

* non-farm employment (as in wage and employment polarization graphs)
drop if occ1990dd>=473 & occ1990dd<=498
foreach y in 1970 1980 1990 2000 2005 {
   egen tot_emp`y'=total(sh_empl`y')
   gen sh_empl`y'nf=sh_empl`y'/tot_emp`y'
}

sort avg_hrwage1980
gen pctile=100*sum(sh_empl1980nf)

quietly gen quintile_1=(pctile<20)
quietly summ pctile if quintile_1==1
quietly gen q1_max=r(max)
quietly summ pctile if quintile_1==0
quietly gen q2_min=r(min)
quietly replace quintile_1=(20-q1_max)/(q2_min-q1_max) if pctile==q2_min

* interact employment shares, service dummies, with quintile variable
foreach y in 1970 1980 1990 2000 2005 {
   quietly egen sh_empl`y'_q1=total(sh_empl`y'nf*quintile_1)
   quietly egen sh_empl`y'_q1_svc=total(sh_empl`y'nf*quintile_1*(occ1_service==1))
   quietly egen sh_empl`y'_q1_nsvc=total(sh_empl`y'nf*quintile_1*(occ1_service==0))
}
* employment share in bottom quintile, svc vs non-svc
foreach y in 1970 1980 1990 2000 2005 {
   disp "emp share by quintile, year `y'"
   summ sh_empl`y'_q1
   summ sh_empl`y'_q1_svc
   summ sh_empl`y'_q1_nsvc
}



log close
