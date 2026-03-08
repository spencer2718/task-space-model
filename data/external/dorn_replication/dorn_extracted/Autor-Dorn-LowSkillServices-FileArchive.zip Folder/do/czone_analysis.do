**************************************************************************************************
*
* Autor and Dorn, "The Growth of Low-Skill Service Jobs and the Polarization of the U.S. Labor Market"
*
* Regressions for Tables 3-6, Table 7 Panel A, Appendix Table 4, and Online Appendix Tables
*
* David Dorn, March 28, 2012
*
**************************************************************************************************


clear
set memory 50m
set more off

capture log close
log using ../log/czone_analysis.log, replace text


use ../dta/workfile2012.dta, clear
xi i.statefip


***** Table 3 - Adjusted PCs per Employee / Share of Employment in Routine Occs *****

reg d_rpc l_sh_routine33a _Istatefip* [aw=timepwt48] if t1==1, cluster(statefip)
reg d_rpc l_sh_routine33a _Istatefip* [aw=timepwt48] if t2==1, cluster(statefip)
reg d_rpc l_sh_routine33a t2 _Istatefip* [aw=timepwt48] if t1==1 | t2==1, cluster(statefip)
reg d_sh_routine33a l_sh_routine33a t2 t3 _Ist* [aw=timepwt48] if yr>=1980, cluster(statefip)
reg d_sh_routine33a_c l_sh_routine33a t2 t3 _Ist* [aw=timepwt48] if yr>=1980, cluster(statefip)
reg d_sh_routine33a_nc l_sh_routine33a t2 t3 _Ist* [aw=timepwt48] if yr>=1980, cluster(statefip)


***** Table 4 - Svc Employment by Period (first stage results reported in AppTab 3) *****

reg d_shocc1_service_nc l_sh_routine33a _Istatefip* [aw=timepwt48] if t50==1, cluster(statefip)
reg d_shocc1_service_nc l_sh_routine33a _Istatefip* [aw=timepwt48] if t70==1, cluster(statefip)
reg d_shocc1_service_nc l_sh_routine33a _Istatefip* [aw=timepwt48] if t1==1, cluster(statefip)
reg d_shocc1_service_nc l_sh_routine33a _Istatefip* [aw=timepwt48] if t2==1, cluster(statefip)
reg d_shocc1_service_nc l_sh_routine33a _Istatefip* [aw=timepwt48] if t3==1, cluster(statefip)

* descriptives
by yr, sort: summ d_shocc1_service_nc [aw=timepwt48]


***** Table 5 - Svc Employment Stacked *****

* Panel A: OLS, controls in levels
regress d_shocc1_service_nc l_sh_routine33a t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
regress d_shocc1_service_nc l_sh_routine33a l_relsup_highlow t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
regress d_shocc1_service_nc l_sh_routine33a l_popfborn_shof_edulow t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
regress d_shocc1_service_nc l_sh_routine33a l_shind_manuf l_unempl t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
regress d_shocc1_service_nc l_sh_routine33a l_sh_empl_f l_shage_65up t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
regress d_shocc1_service_nc l_sh_routine33a l_sh_minw t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
regress d_shocc1_service_nc l_sh_routine33a l_relsup_highlow l_popfborn_shof_edulow l_shind_manuf l_unempl l_sh_empl_f l_shage_65up l_sh_minw t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)

* Panel B: 2SLS, controls in levels
ivregress 2sls d_shocc1_service_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_service_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) l_relsup_highlow t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_service_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) l_popfborn_shof_edulow t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_service_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) l_shind_manuf l_unempl t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_service_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) l_sh_empl_f l_shage_65up t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_service_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) l_sh_minw t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_service_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) l_relsup_highlow l_popfborn_shof_edulow l_shind_manuf l_unempl l_sh_empl_f l_shage_65up l_sh_minw t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)

* Panel C: 2SLS, controls in first differences
ivregress 2sls d_shocc1_service_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_service_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) d_relsup_highlow t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_service_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) d_popfborn_shof_edulow t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_service_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) d_shind_manuf d_unempl t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_service_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) d_sh_empl_f d_shage_65up t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_service_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) l_sh_minw t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_service_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) d_relsup_highlow d_popfborn_shof_edulow d_shind_manuf d_unempl d_sh_empl_f d_shage_65up l_sh_minw t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)


***** Table 6 - Alternative Hypotheses *****

reg d_shocc1_service_nc l_task_std_offshore _Ist* t2 t3 [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_service_nc l_task_std_offshore (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) _Ist* t2 t3 [aw=timepwt48] if yr>=1980, cluster(statefip)
reg d_shocc1_service_nc d_ln90wlsftfy _Ist* t2 t3 [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_service_nc d_ln90wlsftfy (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) _Ist* t2 t3 [aw=timepwt48] if yr>=1980, cluster(statefip)
reg d_shocc1_service_nc d_avgls2080_edu_coll _Ist* t2 t3 [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_service_nc d_avgls2080_edu_coll (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) _Ist* t2 t3 [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_service_nc d_avgls2080_edu_coll_m (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) _Ist* t2 t3 [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_service_nc d_avgls2080_edu_coll_f (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) _Ist* t2 t3 [aw=timepwt48] if yr>=1980, cluster(statefip)


***** Table 7 (Panel A) - Employment by occupation group and gender *****

ivregress 2sls d_shocc1_service_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_transconstr_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_mgmtproftech_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_clericretail_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_product_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_operator_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)

ivregress 2sls d_shocc1_service_ncm (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_transconstr_ncm (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_mgmtproftech_ncm (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_clericretail_ncm (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_product_ncm (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_operator_ncm (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)

ivregress 2sls d_shocc1_service_ncf (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_transconstr_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_mgmtproftech_ncf (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_clericretail_ncf (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_product_ncf (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_operator_ncf (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)


***** Appendix Table 3 - 2SLS First Stage *****

ivregress 2sls d_shocc1_service_nc (l_sh_routine33a=R33a_50_1950) _Istatefip* [aw=timepwt48] if t50==1, cluster(statefip) first
ivregress 2sls d_shocc1_service_nc (l_sh_routine33a=R33a_50_1970) _Istatefip* [aw=timepwt48] if t70==1, cluster(statefip) first
ivregress 2sls d_shocc1_service_nc (l_sh_routine33a=R33a_50_1980) _Istatefip* [aw=timepwt48] if t1==1, cluster(statefip) first
ivregress 2sls d_shocc1_service_nc (l_sh_routine33a=R33a_50_1990) _Istatefip* [aw=timepwt48] if t2==1, cluster(statefip) first
ivregress 2sls d_shocc1_service_nc (l_sh_routine33a=R33a_50_2000) _Istatefip* [aw=timepwt48] if t3==1, cluster(statefip) first
ivregress 2sls d_shocc1_service_nc (l_sh_routine33a=R33a_50_1950 R33a_50_1970 R33a_50_1980 R33a_50_1990 R33a_50_2000) t70 t1 t2 t3 _Istatefip* [aw=timepwt48], cluster(statefip) first


***** Appendix Table 4 (Panel A) - Detailed Service Occupations *****

ivregress 2sls d_shocc3_food_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc3_janitor_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc3_shealth_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc3_clean_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc3_child_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc3_beauty_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc3_guard_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc3_recreation_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc3_othpers_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)


***** Online Appendix Table 1 - Alternative RSH Measures *****

ivregress 2sls d_shocc1_service_nc (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_service_nc (l_sh_routine33b=R33b_50_1980 R33b_50_1990 R33b_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_service_nc (l_sh_routine33a_sts=R33a_sts_50_1980 R33a_sts_50_1990 R33a_sts_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_service_nc (l_sh_routine33a_finger=R33a_finger_50_1980 R33a_finger_50_1990 R33a_finger_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_service_nc (l_sh_routine33a_nc=R33a_nc_50_1980 R33a_nc_50_1990 R33a_nc_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_service_nc (l_sh_routine25a=R25a_50_1980 R25a_50_1990 R25a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_shocc1_service_nc (l_sh_routine40a=R40a_50_1980 R40a_50_1990 R40a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)

* descriptives: 80th vs 20th percentile of RSH measure
foreach var of varlist l_sh_routine33a l_sh_routine33b l_sh_routine33a_sts l_sh_routine33a_finger l_sh_routine33a_nc l_sh_routine25a l_sh_routine40a {
   _pctile `var' [aw=timepwt48] if yr==1980, p(20, 80)
   disp "20th and 80th percentile of `var'"
   return list
}

***** Online Appendix Table 2 - Educational Composition and Migration *****

ivregress 2sls d_sh_edu_lths (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_sh_edu_hsch (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_sh_edu_scoll (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_sh_edu_coll (l_sh_routine33a=R33a_50_1980 R33a_50_1990 R33a_50_2000) t2 t3 _Istatefip* [aw=timepwt48] if yr>=1980, cluster(statefip)
ivregress 2sls d_sh_mignat_lths (l_sh_routine33a=R33a_50_1980 R33a_50_1990) t2 _Istatefip* [aw=timepwt48] if yr>=1980 & yr<2000, cluster(statefip)
ivregress 2sls d_sh_mignat_hsch (l_sh_routine33a=R33a_50_1980 R33a_50_1990) t2 _Istatefip* [aw=timepwt48] if yr>=1980 & yr<2000, cluster(statefip)
ivregress 2sls d_sh_mignat_scoll (l_sh_routine33a=R33a_50_1980 R33a_50_1990) t2 _Istatefip* [aw=timepwt48] if yr>=1980 & yr<2000, cluster(statefip)
ivregress 2sls d_sh_mignat_coll (l_sh_routine33a=R33a_50_1980 R33a_50_1990) t2 _Istatefip* [aw=timepwt48] if yr>=1980 & yr<2000, cluster(statefip)


log close
