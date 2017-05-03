import argparse
import ast
import pandas
import scipy
import os
import shapefile # https://github.com/GeospatialPython/pyshp
import geocoder
from shapely.geometry import LineString

# **********************
# Read in arguments passed in from Tkinter gui in scrape_gui.py
# **********************
parser = argparse.ArgumentParser() # allow the creation of arguments
namespace,unparsed = parser.parse_known_args() # unpack arguments

def parse_arg(arg):
	k,v = arg.split('=',1) # split over equal sign (ie. extract (q, 'query'))
	try:
		v = ast.literal_eval(v) # evaluate the string as a python literal
	except ValueError:
		pass # if not, evaluate as string
	return k.lstrip('-'),v # parsed argument

d = dict(parse_arg(arg) for arg in unparsed) # create dictionary of arguments

search = d['city'] + ' ' + d['state']# extract search string (ie. 'austin')
hotel = d['base'].replace('_',' ')
form = 'csv' # data extension
state = d['state']

# Retrieve state name from acronym using lookup table
lookup = pandas.read_csv('states_lookup_table.csv')
statename = lookup[lookup.abbreviation == state].name.values[0]

# **********************
# Create nodes dataset from attractions, running scrapy if needed
# **********************
try: 
	nodes = pandas.read_csv(search + '.' + form) # read csv if it already exists
	print 'Existing scrapy .csv output retrieved.'
except IOError:
	# If data does not exist, import crawler and extract data
	from tripadvisor_scraper.run_tripadvisor_spider import TACrawler
	print 'No data to read. TACrawler initiated to collect data.'
	TACrawler(search=search,numdata=1000,dldelay=0,form='csv')
	nodes = pandas.read_csv(search + '.' + form) # read data from csv
	print 'New scrapy .csv output retrieved.'	

# Add hotel to nodes dataframe
hotellatlon = geocoder.google(hotel + ' ' + search).latlng # geocode address
nodes.loc[len(nodes)] = [str(hotellatlon),'hotel','','']

# Retrieve lat and lon values as floats, and write to nodes df
lat = nodes.latlon.str.extract('\[([0-9-.]*),',expand=True) # get latitudes
lon = nodes.latlon.str.extract(', ([0-9-.]*)\]',expand=True) # get longitudes
nodes['lat'] = lat.astype(float).round(9) # round to 6th decimal
nodes['lon'] = lon.astype(float).round(9) # round to 6th decimal

del nodes['latlon']

# **********************
# Convert road network into pandas dataframe
# **********************
def shp2df(fname,fields=False): 
	"""Convert from ESRI shapefile to Pandas dataframe
	from: http://permalink.gmane.org/gmane.comp.python.pydata/1870"""
	shp = shapefile.Reader(fname)
	records = shp.records() # records (rows)
	fld = scipy.array(shp.fields[1:],dtype=str)
	data = pandas.DataFrame(records, columns=fld[:,0])
	del shp
	if fields:
		data = data[[f for f in fields]]
	return data

edge_shp = 'spatial/output/' + d['city'].lower() + '_roads.shp'

try: 
	edges = shp2df(edge_shp,fields=['oneway','fclass','LENGTH_GEO',
		'START_X','START_Y','END_X','END_Y'])
	print 'Existing clipped shapefile retrieved.'
except: 
	# **********************
	# Clip State shapefile to local shapefile
	# **********************
	print 'Clipping new shapefile for local region...'

	# File location for State shapefile to clip from
	shpdest = 'spatial/final/'
	# shpdest = '/media/paul/pman/compopt/roadnetwork/usa/final/'
	roads_shp = shpdest + '{0}_roads_final.shp'.format(statename)
	# roads_shp = 'spatial/TX/texas.shp'

	# Retrieve max and min lat and lon values for clip bounding box
	# Note this only works in North-Eastern quarter of the globe
	# due to positive lat and negative lon values
	left = nodes.min(0,-1)['lon'] # minimum longitude
	bot = nodes.min(0,-2)['lat'] # minimum latitude
	right = nodes.max(0,-1)['lon'] # maximum longitude
	top = nodes.max(0,-2)['lat'] # maximum latitude

	# Add buffer to bounding box values
	buf = 0.1 # Buffer out this many decimal degrees in all directions
	left = left - buf
	bot = bot - buf
	right = right + buf
	top = top + buf

	# ogr2ogr -f "ESRI Shapefile" output.shp input.shp -clipsrc <x_min> <y_min> <x_max> <y_max>
	clip = 'ogr2ogr -f "ESRI Shapefile" ' + edge_shp + ' ' + roads_shp + ' -clipsrc ' + \
			str(left) + ' ' + str(bot) + ' ' + str(right) + ' ' + str(top)
	print clip
	os.system(clip)
	edges = shp2df(edge_shp,fields=['oneway','fclass','LENGTH_GEO',
		'START_X','START_Y','END_X','END_Y'])
	print 'Newly clipped shapefile retrieved'
	print 'Verifying new outputs...'
	# Compare outputs to verify. Generally seem pretty close! 
	print top, max( edges.max(0,3)['START_Y'], edges.max(0,5)['END_Y'] )
	print bot, min( edges.min(0,3)['START_Y'], edges.min(0,5)['END_Y'] )
	print right, max( edges.max(0,2)['START_X'], edges.max(0,4)['END_X'] )
	print left, min( edges.min(0,2)['START_X'], edges.min(0,4)['END_X'] )

edges.columns = ['oneway','fclass','miles','startlon','startlat','endlon','endlat']

# **********************
# Convert latlon points to LINESTRING format for mapping
# **********************
startpoints = []
endpoints = []
for i in range(len(edges)):
	sp = (edges['startlon'].values[i],edges['startlat'].values[i])
	startpoints.append(sp) # shapely pointfile of startpoint
	ep = (edges['endlon'].values[i],edges['endlat'].values[i])
	endpoints.append(ep) # shapely pointfile of endpoint

lines = []
for a,b in zip(startpoints,endpoints):
	l = LineString([a,b]) # line
	lines.append(l.wkt) # add as well-known-text format

edges['LINESTRING'] = lines # add to edges dataset

# **********************
# Create road network
# **********************
import vacation_itinerary_solver
import matplotlib.pyplot as plt

city_itinerary = vacation_itinerary_solver.vacation_itinerary(
	city_file=edges,attractions_file=nodes)
print 'Itinerary class initiated'

optimalItin = city_itinerary.solve_optimal_itinerary(
	itin=city_itinerary.initial_itinerary)
print optimalItin
print 'total reward: ', city_itinerary.getItineraryReward(
	itin=optimalItin)
# city_itinerary.drawStreetNetwork()
# print 'Street network drawn'
# city_itinerary.drawItineraryPath(itin=optimalItin)
# print 'Shortest path drawn'
# city_itinerary.draw_all_attractions(itin=optimalItin)
# print 'Attractions drawn'
city_itinerary.draw_optimal_itinerary(opt_itin=optimalItin,co='orange',
	fname='optimal_itinerary.png',labelsOnOff=True,state_lines=True)
# city_itinerary.zoomToFit(itin=optimalItin,
# 	filename='optimal_itinerary.png')

# **********************
# Display road network and nodes in tkinter gui
# **********************

output_script = 'output.py'
map_file = 'optimal_itinerary.png'

command = str(
	'python ' + output_script
	+ ' -map=' + map_file
	+ ' -city=' + d['city']
	+ ' -state=' + d['state']
	+ ' -base=' + str(hotel).replace(' ','_')
	+ ' -path=' + str(optimalItin).replace(' ','_')
	)

print 'command-line argument:',command

os.system(command)

edges.to_csv('{0}_edges.csv'.format(d['city']),index=False) # for testing
nodes.to_csv('{0}_nodes.csv'.format(d['city']),index=False) # for testing