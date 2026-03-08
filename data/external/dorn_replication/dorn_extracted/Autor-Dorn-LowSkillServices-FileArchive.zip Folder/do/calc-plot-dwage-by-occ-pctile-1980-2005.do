set more 1

version 9.2

/*

  Autor and Dorn, "The Growth of Low-Skill Service Jobs and the Polarization of the U.S. Labor Market"

  Wage Polarization Graphs for Figures 1B, 2B and 5B

  D. Autor, 3/22/2011

  Input data:
  occ-means-by-decade-1980-2005-czall.dta
  occ-means-by-decade-1980-2005-czlow-abovemean-RSHA.dta
  occ-means-by-decade-1980-2005-czhigh-abovemean-RSHA.dta
  dwg-byperc-1980-2005-czall.dta
  dwg-byperc-1980-2005-czlow-abovemean-RSHA.dta
  dwg-byperc-1980-2005-czhigh-abovemean-RSHA.dta

  Output graphs:
  dhrwg-3dec-occ100-czall-color.gph
  dhrwg-8005-cntr-occ100-czall-color.gph
  dhrwg-occ100-8005-abovemean-RSHA-hilow-color.gph

*/

clear
set mem 10m

* Set save variable
if !(_caller()<10) {
   local savemode = "old"
   disp "STATA FILE SAVING MODE: OLD FILE FORMAT"
}

********************************************************************************************************
** Three loops through the program: All CZs, CZs with RSH > mean, CZs with RSH < mean
** Note that czhigh #must# iterate before czlow to complete final figure
********************************************************************************************************

foreach mainloop in czall czhigh czlow {
	local suffix=""
	if "`mainloop'"!="czall" local suffix="-abovemean-RSHA"

	capture log close
	if "`mainloop'"=="czall" log using ../log/calc-plot-dwage-by-occ-pctile-1980-2005-`mainloop'.log,replace text
	if "`mainloop'"!="czall" log using ../log/calc-plot-dwage-by-occ-pctile-1980-2005-`mainloop'`suffix'.log,replace text

	********************************************************************************************************
	** Part I
	** Prep matrix of occs ranked by wage indicating their employment shares at each percentile in 1980.
	********************************************************************************************************

	* Four passes: One for 1980 occs, one for 1990 occs, one for 2000 occs, one for 2005 occs
	foreach pass in 1980 1990 2000 2005 {

		* Eliminate any occs that don't exist in 1980 but do exist in some other years
		
		use ../dta/occ-means-by-decade-1980-2005-czall if year==1980, clear

		* Assert that no farm occs are present
		assert occ1990dd<473 | occ1990dd>498

		* Assign weights - Reweighting to specified year unless reweight is "none"
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
		keep occ1990dd psh1-psh100
		sort occ1990dd
		gen year=`pass'
		save`savemode' ../dta/emp-meanrank-temp`pass'-`mainloop'`suffix'.dta, replace
	}

	append using ../dta/emp-meanrank-temp1980-`mainloop'`suffix'
	append using ../dta/emp-meanrank-temp1990-`mainloop'`suffix'
	append using ../dta/emp-meanrank-temp2000-`mainloop'`suffix'
	sort year occ1990dd
	save`savemode' ../dta/emp-meanrank-temp1980-2005-`mainloop'`suffix'.dta,replace
	desc
	tab year
	tab occ1990dd
	egen v=rsum(psh*)
	tab year, summ(v)
	assert v>=0.99 & v<=1.01
	drop v
	erase ../dta/emp-meanrank-temp1980-`mainloop'`suffix'.dta
	erase ../dta/emp-meanrank-temp1990-`mainloop'`suffix'.dta
	erase ../dta/emp-meanrank-temp2000-`mainloop'`suffix'.dta
	erase ../dta/emp-meanrank-temp2005-`mainloop'`suffix'.dta

	********************************************************************************************************
	** Part II: Calculate wages by occupational percentile
	********************************************************************************************************

	clear
	set mem 100m
	if "`mainloop'"=="czall" use ../dta/occ-means-by-decade-1980-2005-`mainloop'
	if "`mainloop'"!="czall" use ../dta/occ-means-by-decade-1980-2005-`mainloop'`suffix'
	sort year occ1990dd
	merge year occ1990dd using ../dta/emp-meanrank-temp1980-2005-`mainloop'`suffix'
	tab year _merge
	assert _merge==1 | _merge==3
	keep if _merge==3
	drop _merge
	erase ../dta/emp-meanrank-temp1980-2005-`mainloop'`suffix'.dta

	* Counterfactual service wage levels: 1980 svc level + (post 1980) mean product_operator change
	foreach y in 1980 1990 2000 2005 {
		egen prodoper_wg`y' = max(mn_wg_prodoper*(year==`y'))
		egen occ_wg`y' = max(occ_mn_wg*(year==`y')), by(occ1990dd)
	}
	gen d_prodoper8090 = prodoper_wg1990-prodoper_wg1980
	gen d_prodoper9000 = prodoper_wg2000-prodoper_wg1990
	gen d_prodoper0005 = prodoper_wg2005-prodoper_wg2000
	gen d_prodoper8005 = prodoper_wg2005-prodoper_wg1980
	gen d_prodoper9005 = prodoper_wg2005-prodoper_wg1990
	gen d_prodoper8000 = prodoper_wg2000-prodoper_wg1980
	summ prodoper_wg* d_prodoper*

	gen cntr_tmp_8090_wg = (1-occ_service)*occ_mn_wg + occ_service*(occ_wg1980 + d_prodoper8090)
	gen cntr_tmp_8000_wg = (1-occ_service)*occ_mn_wg + occ_service*(occ_wg1980 + d_prodoper8000)
	gen cntr_tmp_8005_wg = (1-occ_service)*occ_mn_wg + occ_service*(occ_wg1980 + d_prodoper8005)
	gen cntr_tmp_9000_wg = (1-occ_service)*occ_mn_wg + occ_service*(occ_wg1990 + d_prodoper9000)
	gen cntr_tmp_9005_wg = (1-occ_service)*occ_mn_wg + occ_service*(occ_wg1990 + d_prodoper9005)
	gen cntr_tmp_0005_wg = (1-occ_service)*occ_mn_wg + occ_service*(occ_wg2000 + d_prodoper0005)

	** Per request of referee, assume *zero* wage growth for service occs in counterfactual wage exercise
	gen cntr_8090_wg = (1-occ_service)*occ_mn_wg + occ_service*(occ_wg1980)
	gen cntr_8000_wg = (1-occ_service)*occ_mn_wg + occ_service*(occ_wg1980)
	gen cntr_8005_wg = (1-occ_service)*occ_mn_wg + occ_service*(occ_wg1980)
	gen cntr_9000_wg = (1-occ_service)*occ_mn_wg + occ_service*(occ_wg1990)
	gen cntr_9005_wg = (1-occ_service)*occ_mn_wg + occ_service*(occ_wg1990)
	gen cntr_0005_wg = (1-occ_service)*occ_mn_wg + occ_service*(occ_wg2000)

	summ d_prodoper*
	summ cntr_tmp_8005_wg cntr_8005_wg if occ_service
	summ cntr_tmp_8005_wg cntr_8005_wg if !occ_service
	drop cntr_tmp*

	* Weighted sum of wages in cell
	sort year
	forvalues perc=1(1)100 {
		egen percwt=sum(psh`perc'*(occ_mn_wg!=.)),by(year)
		egen percwg_`perc'=sum(psh`perc'*occ_mn_wg/percwt),by(year)
		foreach v in 8090 8000 8005 9000 9005 0005 {
			egen cntr_`v'_`perc'=sum(psh`perc'*cntr_`v'_wg/percwt),by(year)
		}
		drop percwt
	}
	foreach v in 8090 8000 8005 9000 9005 0005 {
		drop cntr_`v'_wg
	}

	* Reduce to year by percentile observations
	bysort year: keep if _n==1
	desc
	summ
	reshape long percwg_ cntr_8090_ cntr_8000_ cntr_8005_ cntr_9000_ cntr_9005_ cntr_0005_, i(year) j(pctile)
	keep year pctile percwg* cntr_*
	rename percwg_ percwg
	foreach v in 8090 8000 8005 9000 9005 0005 {
		rename cntr_`v'_ cntr_`v'
	}
	sort pctile year
	bysort pctile: gen dpercwg = percwg - percwg[_n-1]
	bysort pctile: gen d2percwg = percwg - percwg[_n-2]
	bysort pctile: gen d3percwg = percwg - percwg[_n-3]
	tab pctile year, summ(dpercwg)

	bysort pctile: gen dcntrwg = cntr_8090 - percwg[_n-1] if year==1990
	bysort pctile: replace dcntrwg = cntr_9000 - percwg[_n-1] if year==2000
	bysort pctile: replace dcntrwg = cntr_0005 - percwg[_n-1] if year==2005

	bysort pctile: gen d2cntrwg = cntr_8000 - percwg[_n-2] if year==2000
	bysort pctile: replace d2cntrwg = cntr_9005 - percwg[_n-2] if year==2005

	bysort pctile: gen d3cntrwg = cntr_8005 - percwg[_n-3] if year==2005

	bysort year: summ d* d2* d3*
	drop cntr_*

	* Filter out percentiles where we are missing data
	egen zeros=sum(percwg==0),by(pctile)
	tab zeros
	assert zeros!=.

	********************************************************************************************************
	** Part III: Make plots
	********************************************************************************************************

	* Reshape to by-decade changes
	reshape wide percwg dpercwg d2percwg d3percwg dcntrwg d2cntrwg d3cntrwg, i(pctile) j(year)
	summ d* d2* d3*
	assert dpercwg1980==.
	drop dpercwg1980
	rename dpercwg1990 dpercwg8090
	rename dpercwg2000 dpercwg9000
	rename dpercwg2005 dpercwg0005

	rename d2percwg2005 dpercwg9005
	rename d2percwg2000 dpercwg8000
	rename d3percwg2005 dpercwg8005

	rename dcntrwg1990 dcntrwg8090
	rename dcntrwg2000 dcntrwg9000
	rename dcntrwg2005 dcntrwg0005

	rename d2cntrwg2005 dcntrwg9005
	rename d2cntrwg2000 dcntrwg8000
	rename d3cntrwg2005 dcntrwg8005


	local cztxt = "Smoothed Changes in Real Hourly Wages by Skill Percentile"
	local cztxt2 = "Observed and Counterfactual Changes in Hourly Wages by Skill Percentile"
	if "`mainloop'"=="czall" local subtit ""
	if "`mainloop'"=="czlow" {
		local subtit "Commuting zones with Below Mean Routine Share in 1980"
	}
	if "`mainloop'"=="czhigh" {
		local subtit "Commuting Zones with Above Mean Routine Share in 1980"
	}

	label var dpercwg8090 "1980-1990"
	label var dpercwg9000 "1990-2000"
	label var dpercwg0005 "2000-2005"
	label var dpercwg9005 "1990-2005"
	label var dpercwg8000 "1980-2000"
	label var dpercwg8005 "1980-2005"

	label var dcntrwg8090 "1980-1990"
	label var dcntrwg9000 "1990-2000"
	label var dcntrwg0005 "2000-2005"
	label var dcntrwg9005 "1990-2005"
	label var dcntrwg8000 "1980-2000"
	label var dcntrwg8005 "1980-2005"

	summ dpercwg*
	label var pctile "Occupation's Percentile in 1980 Wage Distribution"

	sort pctile
	foreach yr in 8090 9000 0005 9005 8000 8005 {
		lowess dpercwg`yr' pctile if zeros==0 , gen(pdperc`yr') bwidth(.4) nograph
		lowess dcntrwg`yr' pctile if zeros==0 , gen(pdcntr`yr') bwidth(.4) nograph
	}
	label var pdperc8090 "Observed 1980-1990"
	label var pdperc9000 "Observed 1990-2000"
	label var pdperc0005 "Observed 2000-2005"
	label var pdperc9005 "Observed 1990-2005"
	label var pdperc8000 "Observed 1980-2000"
	label var pdperc8005 "Observed 1980-2005"

	label var pdcntr8090 "Wage growth service occs = Zero"
	label var pdcntr9000 "Wage growth service occs = Zero"
	label var pdcntr0005 "Wage growth service occs = Zero"
	label var pdcntr9005 "Wage growth service occs = Zero"
	label var pdcntr8000 "Wage growth service occs = Zero"
	label var pdcntr8005 "Wage growth service occs = Zero"

	if "`mainloop'"=="czall" {

		* Figure for paper: Change in mean wage by skill percentile 1980-2005
		* set scheme s2mono
		* tw scatter pdperc8005 pctile if pctile>=5 & pctile<=95 &  zeros==0, connect( l)  msymbol(d) msize(small) ylabel(0.05(.05)0.3) title("`cztxt' 1980-2005", size(medium)) subtitle("`subtit'", size(medsmall)) xtitle("Skill Percentile (Ranked by 1980 Occupational Mean Wage)", size(medium))  l1title("Change in Real Log Hourly Wage", size(medsmall)) ytitle("") saving(../gph/dhrwg-3dec-occ100-`mainloop'`suffix'-bw.gph, replace)

		set scheme s2color
		tw scatter pdperc8005 pctile if pctile>=5 & pctile<=95 &  zeros==0, connect( l)  msymbol(d) msize(small) ylabel(0.05(.05)0.3) title("`cztxt' 1980-2005", size(medium)) subtitle("`subtit'", size(medsmall)) xtitle("Skill Percentile (Ranked by 1980 Occupational Mean Wage)", size(medium))  l1title("Change in Real Log Hourly Wage", size(medsmall)) ytitle("") saving(../gph/dhrwg-3dec-occ100-`mainloop'`suffix'-color.gph, replace)

		* Figure for paper: Change in mean wage by skill percentile 1980-2005, observed and counterfactual
		* set scheme s2mono
		* tw scatter pdperc8005 pdcntr8005 pctile if pctile>=5 & pctile<=95 &  zeros==0, connect(l l)  lpattern(solid dash) msymbol(sh i) msize(small small) ylabel(0.05(.05)0.30) title("`cztxt2' 1980-2005", size(medium)) subtitle("`subtit'", size(medsmall)) xtitle("Skill Percentile (Ranked by 1980 Occupational Mean Wage)", size(medium)) l1title("Change in Real Log Hourly Wage", size(medsmall)) ytitle("") saving(../gph/dhrwg-8005-cntr-occ100-`mainloop'`suffix'-bw.gph, replace)

		set scheme s2color
		tw scatter pdperc8005 pdcntr8005 pctile if pctile>=5 & pctile<=95 &  zeros==0, connect(l l)  lpattern(solid dash) msymbol(sh i) msize(small small) ylabel(0.05(.05)0.30) title("`cztxt2' 1980-2005", size(medium)) subtitle("`subtit'", size(medsmall)) xtitle("Skill Percentile (Ranked by 1980 Occupational Mean Wage)", size(medium)) l1title("Change in Real Log Hourly Wage", size(medsmall)) ytitle("") saving(../gph/dhrwg-8005-cntr-occ100-`mainloop'`suffix'-color.gph, replace)

	}

	* Save data for pooling with high-low CZ's for comparison
	keep pctile pdperc* pdcntr* zeros dpercwg*
	rename zeros `mainloop'_zeros
	renpfix pdperc `mainloop'_pdperc
	renpfix pdcntr `mainloop'_pdcntr
	renpfix dpercwg `mainloop'_dpercwg
	sort pctile
	desc, f
	summ
	save`savemode' ../dta/dwg-byperc-1980-2005-`mainloop'`suffix'.dta, replace

	********************************************************************************************************
	** Part IV: Plots pooling high and low-czones
	** This can only run after both high and low plots have run
	** We assume that czhigh runs first, so if we get to czlow, the necessary czhigh data will be present
	********************************************************************************************************

	if "`mainloop'"=="czlow" {
		clear
		use ../dta/dwg-byperc-1980-2005-czlow`suffix'.dta
		merge pctile using ../dta/dwg-byperc-1980-2005-czhigh`suffix'.dta
		assert _merge==3
		drop _merge

		local cztxt = "Smoothed Changes in Real Hourly Wages by Occupational Skill Percentile"
		local cztxtm = "Smoothed Changes in Hourly Wages Relative to Median by Occupational Skill Percentile"

		local subtit "Commuting Zones Split on Mean Routine Share in 1980"

		local yr = "8005"
		local cztxt1 = "`cztxt' 1980-2005"
		local cztxt2 = "`cztxtm' 1980-2005"

		label var czlow_pdperc`yr' "Low Routine Share"
		label var czhigh_pdperc`yr' "High Routine Share"

		* set scheme s2mono
		* tw scatter czlow_pdperc`yr' czhigh_pdperc`yr' pctile if pctile>=5 & pctile<=95, connect(l l) lpattern(solid dash) msymbol(sh i) msize(small small) ylabel(.10(.05)0.30) title("`cztxt1'", size(medium)) subtitle("`subtit'", size(medsmall)) xtitle("Skill Percentile (Ranked by 1980 Occupational Mean Wage)", size(medium)) l1title("Change in Real Log Hourly Wage", size(medsmall)) ytitle("") saving(../gph/dhrwg-occ100-`yr'`suffix'-hilow-bw.gph, replace)

		set scheme s2color
		tw scatter czlow_pdperc`yr' czhigh_pdperc`yr' pctile if pctile>=5 & pctile<=95, connect(l l) lpattern(solid dash) msymbol(sh i) msize(small small) ylabel(.10(.05)0.30) title("`cztxt1'", size(medium)) subtitle("`subtit'", size(medsmall)) xtitle("Skill Percentile (Ranked by 1980 Occupational Mean Wage)", size(medium)) l1title("Change in Real Log Hourly Wage", size(medsmall)) ytitle("") saving(../gph/dhrwg-occ100-`yr'`suffix'-hilow-color.gph, replace)

	}

}
