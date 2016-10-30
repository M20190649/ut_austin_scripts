from bs4 import BeautifulSoup
# import requests
import urllib

url = 'https://www.google.com/#q=austin+capitol'

def linkslist(url):
	urlfile = urllib.urlopen(url)
	urllines = urlfile.readlines()
	links = []
	for i in range(len(urllines)): 
		soup = BeautifulSoup(urllines[i], 'html.parser')
		print soup.find_all('body')
	# 	for link in soup.find_all('a'): 
	# 		current = link.get('href')
	# 		links.append(current)
	# print links

linkslist(url)