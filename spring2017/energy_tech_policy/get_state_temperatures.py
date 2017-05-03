import pandas
import requests
import lxml.html
import re
import csv

url = 'https://www.ncdc.noaa.gov/cag/time-series/us/{0}/0/{1}/12/12/2010-2011?base_prd=true&firstbaseyear=1901&lastbaseyear=2000'

options = ['pcp','tavg','tmax','tmin']

# states = pandas.read_csv('state_to_abbrv.csv').Abbreviation.unique()

# CONUS states only (AK, HI, DC, & PR excluded)
states = ['AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL',
	'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
	'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
	'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
	'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']

items = [x for x in enumerate(states,1)]

results = []

with open('uscd_state_data_2010_conus.csv','wb') as f:
	w = csv.writer(f)
	w.writerow(['State','NOAA_Precipitation_Annual_Average_Inches', 
		'NOAA_Temperature_Annual_Average_Farenheit',
		'NOAA_Temperature_Annual_Maximum_Farenheit',
		'NOAA_Temperature_Annual_Minimum_Farenheit'])
	for i in items:
		cur_res = []
		cur_res.append(i[1])
		for o in options:
			new_url = url.format(i[0],o)
			page = requests.get(new_url)
			tree = lxml.html.fromstring(page.content)
			value = tree.xpath('//*[@id="values"]/tbody/tr[1]/td[2]/text()')[0]
			res = str(re.findall("\d+\.\d+",value)[0])
			# res = str(value.strip(u'\N{DEGREE SIGN}F')) # Remove degree-F and u'
			cur_res.append(res)
		w.writerow(cur_res)
		print '{0} finished: {1}/{2}'.format(i[1],i[0],len(states))