set more off
clear

// To begin with, note that 'xlong' and 'yhat' data are from matlab data91.txt.
// (given in Ertur and Koch from online open codes)

local chunk=1
local max_iter=1
local t1=1950
local t2=1990


* = Geographic distance ======================================================
clear
infile str3 countryisocode xlong ylat using geo_dist.csv
drop in 1
keep countryisocode xlong ylat
save geo_dist, replace

* = Neoclassical Growth model data ===========================================
* Period
clear
infile str3 countryisocode year rgdpch rgdpwok kc pop openk xrat rgdpl ki rgdptt using pwt71.csv
drop in 1
keep if year>=`t1' & year<=`t2'

* Keep countries
keep if (countryisocode=="NZL") | (countryisocode=="AUS") | (countryisocode=="PNG") | (countryisocode=="IDN") | (countryisocode=="SGP") | ///
(countryisocode=="MYS") | (countryisocode=="LKA") | (countryisocode=="THA") | (countryisocode=="PHL") | (countryisocode=="HKG") | ///
(countryisocode=="BGD") | (countryisocode=="NPL") | (countryisocode=="IND") | (countryisocode=="ISR") | (countryisocode=="JOR") | ///
(countryisocode=="SYR") | (countryisocode=="PAK") | (countryisocode=="JPN") | (countryisocode=="KOR") | (countryisocode=="MOZ") | ///
(countryisocode=="ZAF") | (countryisocode=="BWA") | (countryisocode=="MUS") | (countryisocode=="MDG") | (countryisocode=="ZWE") | ///
(countryisocode=="ZMB") | (countryisocode=="MWI") | (countryisocode=="AGO") | ///
(countryisocode=="TZA") | (countryisocode=="ZAR") | ///
(countryisocode=="COG") | (countryisocode=="BDI") | (countryisocode=="RWA") | (countryisocode=="KEN") | (countryisocode=="UGA") | ///
(countryisocode=="CMR") | (countryisocode=="CAF") | (countryisocode=="CIV") | (countryisocode=="GHA") | (countryisocode=="TGO") | ///
(countryisocode=="NGA") | (countryisocode=="BEN") | (countryisocode=="SLE") | (countryisocode=="ETH") | (countryisocode=="TCD") | ///
(countryisocode=="BFA") | (countryisocode=="MLI") | (countryisocode=="NER") | (countryisocode=="SEN") | (countryisocode=="MRT") | ///
(countryisocode=="EGY") | (countryisocode=="MAR") | (countryisocode=="TUN") | (countryisocode=="GRC") | (countryisocode=="PRT") | ///
(countryisocode=="TUR") | (countryisocode=="ESP") | (countryisocode=="ITA") | (countryisocode=="CHE") | (countryisocode=="AUT") | ///
(countryisocode=="FRA") | (countryisocode=="BEL") | (countryisocode=="GBR") | (countryisocode=="NLD") | (countryisocode=="IRL") | ///
(countryisocode=="DNK") | (countryisocode=="SWE") | (countryisocode=="NOR") | (countryisocode=="FIN") | (countryisocode=="URY") | ///
(countryisocode=="ARG") | (countryisocode=="CHL") | (countryisocode=="PRY") | (countryisocode=="BOL") | (countryisocode=="BRA") | ///
(countryisocode=="PER") | (countryisocode=="ECU") | (countryisocode=="COL") | (countryisocode=="PAN") | (countryisocode=="CRI") | ///
(countryisocode=="VEN") | (countryisocode=="TTO") | (countryisocode=="NIC") | (countryisocode=="SLV") | (countryisocode=="HND") | ///
(countryisocode=="GTM") | (countryisocode=="JAM") | (countryisocode=="DOM") | (countryisocode=="MEX") | (countryisocode=="USA") | ///
(countryisocode=="CAN")

sort countryisocode year

* Variables
// y: lny
gen lny=log(rgdpwok) // logged Real income per worker

// x: lns
replace ki=ki/100
gen lns=log(ki)

// x: lnngd
egen id=group(countryisocode), label
label val id
gen nwok=rgdpch*pop/rgdpwok // the number of workers (Caselli, 2005)
bysort id (year): gen n=(nwok[_n]-nwok[_n-1])/nwok[_n-1]
gen gd=0.05
egen ngd=rowtotal(n gd)
gen lnngd=log(ngd)

// z: trade
gen trade=openk*rgdpl*pop // total
gen ltrade=log(trade)

// Keep
keep countryisocode id year lny lns lnngd ltrade

* Data process
// Merge
merge m:1 countryisocode using geo_dist
keep if _merge==3
drop _merge

// Fill in missing values with the latest one (count 1)
local iter=1
while `iter'<=`max_iter' {
	dis `iter'
	foreach x in lny lns lnngd ltrade {
		replace `x'=`x'[_n+1] if `x'==.
	}
	local iter=`iter'+1	
}

// Drop if missing
drop if lny==.

// Average by N years
gen period = `chunk' * floor(year/`chunk')
egen groupid = group(countryisocode period)

foreach x in lny lns lnngd ltrade {
	bysort groupid: egen avg`x'=mean(`x')
}

bysort groupid: gen flag=_n
keep if flag==1

foreach y in lny lns lnngd ltrade {
	drop `y'
	rename avg`y' `y'
}

* - Panel structure ------------------------------------------------
xtset id year
xtdes

// Leave only balanced ones
by id: gen nyear=[_N]
keep if nyear==`t2'-`t1'+1
xtdes
//kdensity ltrade, title(Case 1)
//kdensity lny, title(Case 1)

* = Save =====================================================================
// Save
drop flag period groupid nyear

// To check auto-correlation
sort id year
outsheet countryisocode year lny using data_autocorr.csv, comma replace

// Data for analysis
sort year id
drop id
outsheet countryisocode year xlong ylat lny lns lnngd ltrade using data_analysis.csv, comma replace