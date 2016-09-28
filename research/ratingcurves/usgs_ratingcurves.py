import urllib
import re
import csv
import numpy

output = "/home/paul/mystuff/school/research/usgs_gages.csv"

url = 'http://waterdata.usgs.gov/nwisweb/get_ratings?file_type=exsa&site_no={0}'

def simplify_gages(file):
	all_ids = []
	with open(file, 'r') as f:
		lines = csv.reader(f,delimiter='\t')
		for line in lines:
			if line[0] == 'USGS': 
				if line[1] not in all_ids: 
					if len(str(line[1])) == 8:
						all_ids.append(line[1])

	f.close()
	print 'All ids read'
	return all_ids

def find_relevant_gages(all_ids, url): 
	ids = []
	for i in range(len(all_ids)):
		urlfile = urllib.urlopen(url.format(str(all_ids[i])))
		urllines = urlfile.readlines()
		if not re.search('[a-zA-Z]',urllines[-1]): # No letters
			ids.append(all_ids[i])
			print '{0} of {1} appended'.format(i,len(all_ids))
	print 'Relevant ids found'
	return ids

def write_ids_to_file(file, ids): 
	print 'Currently writing to file...'
	with open(file, 'w') as f:
		for i in ids: 
			f.write(str(i)+'\n')
	print 'File written'
	f.close()

all_ids = simplify_gages('usgsgages_all.csv')

ids = find_relevant_gages(all_ids, url)

write_ids_to_file('usgsgages.txt', ids)