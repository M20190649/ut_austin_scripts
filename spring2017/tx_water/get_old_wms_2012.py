# import pandas
# import requests
# import lxml.html
# import re
# import csv
import urllib2
from bs4 import BeautifulSoup
import codecs

url = 'https://www2.twdb.texas.gov/apps/db12/searchresults_wms.asp?txtText=&lstDbProjId=&lstSelected=&lstCounty=&chkRegion=yes&lstRegion={0}&lstWMSType=&lstWMSSource=&lstWMSWQI=&lstWMSInfra=&lstWMSDate=&Submit=+Submit+'

# options = [chr(i) for i in range(ord('a'),ord('p')+1)]

url_tweak = url.format('p')

response = urllib2.urlopen(url_tweak)

html = response.read()

soup = BeautifulSoup(html,'lxml')

tables = soup.find_all('table')

td_tags = [table.find_all('tr') for table in tables]

print td_tags

table_body = table.find('tbody')

print table_body

1/0

results = []

with open('uscd_state_data_2010_conus.csv','wb') as f:
	w = csv.writer(f)
	w.writerow(['State','NOAA_Precipitation_Annual_Average_Inches', 
		'NOAA_Temperature_Annual_Average_Farenheit',
		'NOAA_Temperature_Annual_Maximum_Farenheit',
		'NOAA_Temperature_Annual_Minimum_Farenheit'])
	for i in items:
		new_url = url.format(i[0],o)
		page = requests.get(new_url)
		tree = lxml.html.fromstring(page.content)
		value = tree.xpath('//*[@id="values"]/tbody/tr[1]/td[2]/text()')[0]
		res = str(re.findall("\d+\.\d+",value)[0])
		# res = str(value.strip(u'\N{DEGREE SIGN}F')) # Remove degree-F and u'
		cur_res.writerow(res)

	print '{0} finished: {1}/{2}'.format(i[1],i[0],len(states))