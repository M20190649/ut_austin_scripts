import pandashp
from shapely.geometry import LineString

edge_shp = 'spatial/output/tuscaloosa_roads.shp'

edges = pandashp.read_shp(edge_shp)

fields=['oneway','fclass','LENGTH_GEO','START_X','START_Y','END_X','END_Y']

edges = edges[[f for f in fields]]

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

print len(lines)

edges['LINESTRING'] = lines # add to edges dataset

print edges

# edges.to_csv('tuscaloosa_test.csv')