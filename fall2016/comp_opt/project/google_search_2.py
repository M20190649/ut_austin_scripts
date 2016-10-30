import requests
import re
from bs4 import BeautifulSoup
# from fake_useragent import UserAgent

# url
url = 'https://www.tripadvisor.com/Attractions'

# parameters in payload
payload = { 'q' : 'austin' }

# setting user-agent
# my_headers = { 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36' }

# response as object
r = requests.get( url, params = payload )#, headers = my_headers ) 

print r
# read and print response with utf-8 encoding
# print( r.text.encode('utf-8') )

# create bs4 object of response r parsed as html
soup = BeautifulSoup( r.text, 'html.parser' )
print soup

# get all h3 tags with class 'r'
h3tags = soup.find_all( 'h3', class_='r' )
# print h3tags

# find url inside each h3 tag using regex
# if found, print; else, ignore exception
for h3 in h3tags: 
	print h3.find_all('a')[0].get('href')
	# try: 
	# 	# print h3.get('href')
	# 	pass
	# except: 
	# 	continue