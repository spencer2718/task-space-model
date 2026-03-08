**************************************************************************************************
*
* Autor and Dorn, "The Growth of Low-Skill Service Jobs and the Polarization of the U.S. Labor Market"
*
* Plots for Figure 6
*
* David Dorn, March 28, 2012
*
**************************************************************************************************

clear
set memory 20m
set more off

capture log close
log using ../log/czone_plot.log, replace text
use ../dta/workfile2012.dta, clear

* 25-year change in service occupation share
by czone, sort: egen d_shocc_service_long=total(d_shocc1_service_nc*t1+d_shocc1_service_nc*t2+0.5*d_shocc1_service_nc*t3)
keep if yr==1980
assert _N==722

* Regressions
reg d_shocc_service_long l_sh_routine33a [aw=timepwt48]
reg d_shocc_service_long l_sh_routine33a [aw=timepwt48] if l_popcount>=750000==1

* Labels
label var l_sh_routine33a "Share of Employment in Routine-Intensive Occs in 1980"
label var d_shocc_service_long "Change in Non-College Service Empl Share"

* Plots
#delimit ;
set scheme s2color;
twoway lfitci d_shocc_service_long l_sh_routine33a [aw=timepwt48], stdp estopts(cluster(statefip)) ||
scatter d_shocc_service_long l_sh_routine33a,
sort msize(small) ylabel(-.05(.05).20) xlabel(0.2(0.05)0.4)
t1("Change in Non-College Service Emp Share by CZ 1980-2005")
xtitle("Share of Employment in Routine-Intensive Occs in 1980")
ytitle("Change in Non-College Service Emp Share")
msymbol(circle)
saving(../gph/czone_plot_svc_r33a.gph, replace);
#delimit cr

#delimit ;
twoway lfitci d_shocc_service_long l_sh_routine33a [aw=timepwt48] if l_popcount>=750000, stdp estopts(cluster(statefip)) ||
scatter d_shocc_service_long l_sh_routine33a if l_popcount>750000,
sort msize(small) ylabel(.01(.02).14) xlabel(0.29(0.02)0.39)
t1("Change in Non-College Service Emp Share by CZ 1980-2005")
xtitle("Share of Employment in Routine-Intensive Occs in 1980")
ytitle("Change in Non-College Service Emp Share")
msymbol(circle) mlabel(city) mlabsize(tiny) mlabcolor(black) mlabposition(6)
saving(../gph/czone_plot_svc_r33a_750k.gph, replace);
#delimit cr

log close
clear
