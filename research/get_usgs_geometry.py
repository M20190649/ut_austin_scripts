import pandas
import scipy

import urllib
import re

weburl = 'https://waterdata.usgs.gov/tx/nwis/measurements?site_no={0}&agency_cd=USGS&format=rdb_expanded'
usgsids = ['08158810','08158860','08158700','08158827','08158970','08158930',
	'08158840','08158927','08159000']

def get_usgs_geometry():
	""" Initializes self.usgsq and self.usgsh """
	usgsh = []
	usgsq = []
	for usgsid in usgsids:
		urlfile = urllib.urlopen(weburl.format(str(usgsid)))
		urllines = urlfile.readlines()

		headers = []
		header_check = False

		# chan_disch = scipy.array([])
		chan_width = scipy.array([])
		# chan_area = scipy.array([])
		# chan_veloc = scipy.array([])
		gage_height = scipy.array([])

		for j in range(len(urllines)):
			line = urllines[j]

			if len(headers) != 0:
				current = line.split('\t')
				# chan_disch = scipy.append( chan_disch, (current[23]) )
				chan_width = scipy.append( chan_width, (current[24]) )
				# chan_area = scipy.append( chan_area, (current[25]) )
				# chan_veloc = scipy.append( chan_veloc, (current[26]) )
				gage_height = scipy.append( gage_height, current[8])

			if line[0] != '#' and header_check == False:
				headers = line.split('\t') # get column headings
				header_check = True

		# Remove 's' stuff usgs puts on their documents (ie. 15s, 2s, etc.)
		# chan_disch = chan_disch[1:]
		chan_width = chan_width[1:] # feet
		# chan_area = chan_area[1:]
		# chan_veloc = chan_veloc[1:]
		gage_height = gage_height[1:] # feet

		gage_height = [x for x in gage_height if x]

		# print gage_height
		# print min(gage_height)
		print 'min: {0}, max: {1}'.format(str(min(gage_height)),str(max(gage_height)))

		# print chan_disch
		# print chan_width
		# print chan_area
		# print chan_veloc

get_usgs_geometry()