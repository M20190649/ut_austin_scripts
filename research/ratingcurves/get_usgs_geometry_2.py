import urllib

weburl = 'https://waterdata.usgs.gov/tx/nwis/measurements?site_no={0}&agency_cd=USGS&format=rdb_expanded'

usgsid = '08158810'

def get_usgs_geometry(usgsid):
	""" Retrieves USGS geometry data """

	# Retrieve data
	urlfile = urllib.urlopen(weburl.format(str(usgsid)))
	urllines = urlfile.readlines()
	urllines = [line.split('\t') for line in urllines if line[0] != '#'] # Ignore details at beginning
	del urllines[1] # Remove additional unnecessary details

	# Separate headers and data
	keys = urllines[0]
	values = urllines[1:]

	d = {k:list(v) for k,v in zip(keys,zip(*values))}
	return d

d = get_usgs_geometry(usgsid)

print d.keys()
# print d['gage_height_va']