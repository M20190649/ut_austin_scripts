import urllib2

url = 'ftp://USGS_P133:ws54R7@ftp.horizon-systems.com/HRNHDPlusBuildRefresh_12xx/'

files = urllib2.urlopen(url).read().splitlines()

# print files[2]

# print files[2].split(' ')

files = [i.split(' ')[-1] for i in files[2:]]

print files


# Download files
import shutil
from contextlib import closing

for f in files:
	with closing(urllib2.urlopen(url + f)) as r:
	    with open('file', 'wb') as f:
	        shutil.copyfileobj(r, f)