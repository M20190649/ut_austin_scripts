import urllib2
import re
from bs4 import BeautifulSoup

usgsid = '08158000'

url = 'https://waterdata.usgs.gov/tx/nwis/measurements?site_no={0}&agency_cd=USGS'

xpathselector = '//*[contains(concat( " ", @class, " " ), concat( " ", "leftsidetext", " " ))]//div[(((count(preceding-sibling::*) + 1) = 6) and parent::*)]'

response = urllib2.urlopen(url.format(usgsid))

soup = BeautifulSoup(response, 'lxml')

print soup.p["class='leftsidetext'"]

1/0

print soup
1/0

htmlparser = etree.HTMLParser()

tree = etree.parse(response, htmlparser)

result = tree.xpath(xpathselector)

print [e.text_content() for e in result]