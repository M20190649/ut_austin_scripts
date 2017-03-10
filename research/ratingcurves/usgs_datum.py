import urllib2
from lxml import etree

usgsid = '08167000'

url = 'https://waterdata.usgs.gov/tx/nwis/measurements?site_no={0}&agency_cd=USGS'
response = urllib2.urlopen(url.format(usgsid))
tree = etree.parse(response, etree.HTMLParser())

xpath_datum = '//*[contains(concat( " ", @class, " " ), concat( " ", "leftsidetext", " " ))]//div[(((count(preceding-sibling::*) + 1) = 6) and parent::*)]'
result_datum = tree.xpath(xpath_datum)
items_datum = [e.text for e in result_datum][0].split()
print items_datum

# Retrieve datum and value
for i in items_datum:
	if i == items_datum[-1]:
		datum_name = i
	try: 
		datum_value = float(i.replace(',', ''))
	except ValueError: 
		continue

print datum_value
print datum_name

# Retrieve latitude and longitude of gage
# Lat = 30.0 + 9.0/60 + 19.0/3600
# Lon = 97.0 + 56.0/60 + 23.0/3600

# Convert from NGDV29 to NADV88
VERTCON = 'https://www.ngs.noaa.gov/cgi-bin/VERTCON/vert_con.prl'
