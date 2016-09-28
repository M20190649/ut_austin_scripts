import urllib
from bs4 import BeautifulSoup
import os.path

def linkslist(url):
	urlfile = urllib.urlopen(url)
	urllines = urlfile.readlines()
	links = []
	for i in range(len(urllines)): 
		soup = BeautifulSoup(urllines[i], 'html.parser')
		for link in soup.find_all('a'): 
			current = link.get('href')
			links.append(current)
	return links

def download_links(links, extensions): 
	dlfiles = []
	for ext in extensions: 
		negext = -len(ext)
		for i in range(len(links)): 
			current = links[i]
			if ext == str(current)[negext:]:
				dlfiles.append(links[i])
	return dlfiles

def filenames(dlfiles):
	filenames = []
	for x in range(len(dlfiles)):
		for y in range(len(dlfiles[x])):
			current = dlfiles[x]
			if str(current)[-y] == "/":
				filenames.append(str(current)[(-y+1):])
				break
	return filenames

def webdl(url,extensions,output):
	urlfile = urllib.urlopen(url)
	urllines = urlfile.readlines()
	links = linkslist(url)
	dlfiles = download_links(links,extensions)
	files = filenames(dlfiles)
	urlsnip = links[0]
	for item in range(len(dlfiles)): 
		thisitem = dlfiles[item]
		currenturl = urlsnip + "/" + thisitem
		print currenturl
		urltowrite = urllib.urlopen(currenturl).read()
		dllink = urllib.urlopen(currenturl)
		output_file = os.path.normcase(output + "%s" % files[item])
		with open(output_file,"wb") as endfile: 
			endfile.write(urltowrite)
	print "Downloaded files to %s" % os.path.normcase(output)

# webdl("http://neddimitrov.org/teaching/201604CO",[".ipynb"],"/home/paul/myscripts/courses/fall2016/comp_opt/old_files/")