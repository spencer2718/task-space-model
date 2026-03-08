set more 1
clear
set mem 10m

/*

  Autor and Dorn, "The Growth of Low-Skill Service Jobs and the Polarization of the U.S. Labor Market"

  Figure 3: Employment Growth in Lowest Skill Quintile

  Input data:
  plotpoints-svc-by-quintile-1970-2005.dta (based on occ1990dd_stats.log)

  Output graph:
  docc-p20-1970-2005.gph

*/


clear
use ../dta/plotpoints-svc-by-quintile-1970-2005,clear
set scheme s2color
replace year = "_t7080" if year=="1970-1980"
replace year = "_t8090" if year=="1980-1990"
replace year = "_t9000" if year=="1990-2000"
replace year = "_t0005" if year=="2000-2005"

replace delta=delta*100
reshape wide delta, i(occ) j(year) string

label var delta_t7080 "1970-1980"
label var delta_t8090 "1980-1990"
label var delta_t9000 "1990-2000"
label var delta_t0005 "2000-2005"

replace occ="1. All Occupations" if occ=="all"
replace occ="2. Service Occupations" if occ=="service"
replace occ="3. Non-Service Occupations" if occ=="non-service"


graph bar (asis) delta_t7080 delta_t8090 delta_t9000 delta_t0005, intensity(75) over(occ) /* title("Changes in Employment Shares by Decade") subtitle("Occupations in Lowest Wage Quartile in 1980") */ legend(on) l2title("Change in share of aggregate employment") ylabel(-2(0.5)1.5) /*bar(3,bstyle(dotchart)) bar(2,bstyle(foreground)) bar(1,bstyle(ci2)) bar(4,bstyle(ci))*/ saving(../gph/docc-p20-1970-2005.gph,replace)
