""" 
Module for finding the optimal vacation itinerary schedule based on the savings algorithm.

"""

import scipy
import pandas
import networkx
from geoplotter import *
import matplotlib.pyplot
import itertools

TIME_VALUE_OF_MONEY = 12.
HOURS_AWAY_FROM_HOTEL = 12.
SAMPLE_SIZE = 5

class vacation_itinerary:

	def __init__(self,city_file,attractions_file,**kwargs):
		# reads road network file
		#self.ds = pandas.read_csv(city_file)
		self.ds = city_file
		# reads attraction address file
		#self.attr = pandas.read_csv(attractions_file)
		self.attr = attractions_file
		self.initial_itinerary = ['hotel']
		self.initial_itinerary.extend(self.attr.attraction.values)
		# creates GeoPlotter object
		self.DSmap = GeoPlotter(epsg=4326)
		# creates networkx graph
		self.createNetworkxGraph()


	def identifyStartPoints(self,df,exp):
	    startPts = df.LINESTRING.str.extract('LINESTRING \(([0-9-.]* [0-9-.]*),',expand=exp)
	    return startPts
	        
	def identifyEndPoints(self,df,exp):
	    endPts = df.LINESTRING.str.extract(', ([0-9-.]* [0-9-.]*)\)',expand=exp)
	    return endPts
	    

	# creates the networkx graph for the road network
	def createNetworkxGraph(self): 
        #creates a networkx graph
	    self.gd = networkx.DiGraph()

	    # Sets up edge sets 
	    # F Streets
	    self.Fds = self.ds[self.ds.oneway=='F']
	    self.FstartPTS = self.identifyStartPoints(df=self.Fds,exp=False)
	    self.FendPTS = self.identifyEndPoints(df=self.Fds,exp=False)
	    
	    # T Streets
	    self.Tds = self.ds[self.ds.oneway=='T']
	    self.TstartPTS = self.identifyStartPoints(df=self.Tds,exp=False)
	    self.TendPTS = self.identifyEndPoints(df=self.Tds,exp=False)
	    
	    # B Streets
	    self.Bds = self.ds[self.ds.oneway=='B']
	    self.BstartPTS = self.identifyStartPoints(df=self.Bds,exp=False)
	    self.BendPTS = self.identifyEndPoints(df=self.Bds,exp=False)

	    # Recall, Time = Distance / Speed
	    # Freeway Speed = 65 mph (fclass == Motorway, Motorway Link, Secondary, Secondary Link)
	    # Other Road Speed = 40 mph
	    timeF = [self.Fds.miles.values[a]/65. if (self.Fds.fclass.values[a]=='motorway') or (self.Fds.fclass.values[a]=='motorway_link') or (self.Fds.fclass.values[a]=='secondary') or (self.Fds.fclass.values[a]=='secondary_link') else self.Fds.miles.values[a]/40. for a in range(len(self.Fds.miles.values))]
	    self.F_dict = [dict(time=mi) for mi in timeF]
	    timeT = [self.Tds.miles.values[a]/65. if (self.Tds.fclass.values[a]=='motorway') or (self.Tds.fclass.values[a]=='motorway_link') or (self.Tds.fclass.values[a]=='secondary') or (self.Tds.fclass.values[a]=='secondary_link') else self.Tds.miles.values[a]/40. for a in range(len(self.Tds.miles.values))]
	    self.T_dict = [dict(time=mi) for mi in timeT]
	    timeB = [self.Bds.miles.values[a]/65. if (self.Bds.fclass.values[a]=='motorway') or (self.Bds.fclass.values[a]=='motorway_link') or (self.Bds.fclass.values[a]=='secondary') or (self.Bds.fclass.values[a]=='secondary_link') else self.Bds.miles.values[a]/40. for a in range(len(self.Bds.miles.values))]
	    self.B_dict = [dict(time=mi) for mi in timeB]

	    # add F streets        
	    F_edges = zip(list(self.FstartPTS.values), list(self.FendPTS.values),self.F_dict)
	    self.gd.add_edges_from(F_edges)
	    
	    # add T streets
	    T_edges = zip(list(self.TendPTS.values), list(self.TstartPTS.values),self.T_dict)
	    self.gd.add_edges_from(T_edges) 
	    
	    # add B streets
	    B_edges1 = zip(list(self.BstartPTS.values), list(self.BendPTS.values),self.B_dict)
	    B_edges2 = zip(list(self.BendPTS.values), list(self.BstartPTS.values),self.B_dict)
	    self.gd.add_edges_from(B_edges1) 
	    self.gd.add_edges_from(B_edges2)

	    # gets the connected subgraphs of the networkx graph and finds the subgraph with the largest number of edges
	    gdcon_temp = list(networkx.strongly_connected_component_subgraphs(self.gd,copy=True))
	    temp_edge_length = scipy.array([len(p.edges()) for p in gdcon_temp])
	    idx = scipy.where(temp_edge_length==max(temp_edge_length))[0][0]
	    self.gdcon = gdcon_temp[idx]
	    print 'NetworkX graph created.'
	    print


	# finds the shortest path between two nodes using Dijkstra's algorithm in networkx
	def getSPNetworkx(self,startnode,destnode,GPH):
		return networkx.shortest_path(GPH,source=startnode,target=destnode,weight='time')


	# plots street network using geoplotter
	def drawStreetNetwork(self, states_and_water_filled=False):
		# Draw countries, oceans and states if states_and_water_filled is true
		if states_and_water_filled:
			from matplotlib.patches import Polygon
			from matplotlib.collections import PatchCollection
			from matplotlib.patches import PathPatch
			# draw world
			"""Draws oceans, countries, and states."""
			self.DSmap.drawMapBoundary()
			#self.DSmap.fillContinents()
			#self.DSmap.drawCoastLines(linewidth=0.7)
			self.DSmap.drawCountries(linewidth=1.5)
			self.DSmap.m.readshapefile('st99_d00', name='states', drawbounds=False)

			patches   = []
			for shape in zip(self.DSmap.m.states):
				patches.append( Polygon(scipy.array(shape)[0], True) )

			ax = self.DSmap.getAxes()
			ax.add_collection(PatchCollection(patches, facecolor= 'beige', edgecolor='k', linewidths=1., zorder=2))

		# extracts line segment points
		maplinestemp = self.ds.LINESTRING.str.extract('LINESTRING \(([0-9-.]* [0-9-.]*, [0-9-.]* [0-9-.]*)\)',expand=True)
		maplinestemp.columns = ['TotLine']
		# separates out points and reformats to feed into drawLines from geoplotter
		temp = [r.split(',') for r in maplinestemp.TotLine.values]
		stLines = [[[float(dw) for dw in p.split()] for p in g] for g in temp]		
		self.DSmap.drawLines(lines = stLines,color = 'b',linewidth = 0.3)		
		

	# plots address(es) - addresLon and addressLat can be lists
	def drawAddress(self,addressLon,addressLat,marksz=75,co='g'):
		self.DSmap.drawPoints(lat=addressLat,lon=addressLon,color=co,s=marksz)


	# draws all attractions as a red dot on the map
	def draw_all_attractions(self,itin,closestNode=True,with_labels=True):
		attr_lon = [self.attr[self.attr.attraction==h].lon.values[0] for h in itin]
		attr_lat = [self.attr[self.attr.attraction==h].lat.values[0] for h in itin]
		if closestNode==True:
			nodes_temp = scipy.array([self.findClosestNode(placeLon=self.attr[self.attr.attraction==h].lon.values[0],placeLat=self.attr[self.attr.attraction==h].lat.values[0],GPH=self.gdcon)[1] for h in itin])
			attr_lon = nodes_temp[:,0]
			attr_lat = nodes_temp[:,1]
		self.DSmap.drawPoints(lat=attr_lat[0],lon=attr_lon[0],color='r',s=75)
		self.DSmap.drawPoints(lat=attr_lat[1:-1],lon=attr_lon[1:-1],color='g',s=75)

		# add sequential labels to attractions
		if with_labels:
			labels = ['hotel']
			lb_temp = [b+1 for b in range(len(itin)-1)]
			labels.extend(['{0}'.format(i) for i in lb_temp])
			labels.extend('hotel')
			#plt.subplots_adjust(bottom = 0.1)
			for lab, x, y in zip(labels, attr_lon, attr_lat):
			    self.DSmap.annotate( text=lab, xy = (x, y), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5), arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))


	# draws the path
	def drawPath(self,path,lw=3,col='orange'):
		stLines = [[[float(dw) for dw in g.split()] for g in path]]
		self.DSmap.drawLines(lines = stLines,color = col,linewidth = lw,alpha = 1)


	# draws the itinerary path
	def drawItineraryPath(self,itin,colP='orange'):
		for locidx in range(len(itin)-1):
			startN = self.findClosestNode(placeLon=self.attr[self.attr.attraction==itin[locidx]].lon.values[0], placeLat=self.attr[self.attr.attraction==itin[locidx]].lat.values[0],GPH=self.gdcon)[0]
			endN = self.findClosestNode(placeLon=self.attr[self.attr.attraction==itin[locidx+1]].lon.values[0], placeLat=self.attr[self.attr.attraction==itin[locidx+1]].lat.values[0],GPH=self.gdcon)[0]
			p=self.getSPNetworkx(startnode=startN, destnode=endN, GPH=self.gdcon)
			self.drawPath(path=p,lw=2,col=colP)


	# draw the attractions and paths for the optimal itinerary on the city map
	def draw_optimal_itinerary(self,opt_itin,co='orange',fname=False,labelsOnOff=True,state_lines=False):
		print 'Drawing Optimal Itinerary...'
		# draws the city map
		self.drawStreetNetwork(states_and_water_filled=state_lines)
		# draws path for the optimal itinerary
		self.drawItineraryPath(itin=opt_itin,colP=co)
		# draws the attractions
		self.draw_all_attractions(itin=opt_itin,with_labels=labelsOnOff)
		self.zoomToFit(itin=opt_itin,filename=fname)


	# draw attractions and paths for the optimal itinerary for multiple days on the city map
	def draw_multi_day_optimal_itinerary(self,multi_day_itin,fname=False,onemap=False,labelsOnOff=True,state_lines=False):
		print 'Drawing Map for Multiple Day Itinerary...'
		print

		#draws the paths and attractions
		if onemap:	# draws multi-day paths in one map
			colorP = itertools.cycle(['orange','m','c','black'])#,'y'])
			# creates list of all attractions for the multi-day for zooming
			temp_itin = multi_day_itin[0]
			for u in multi_day_itin[1:]:
				temp_itin.extend(u)
			# draws the city map
			self.drawStreetNetwork(states_and_water_filled=state_lines)
			for w in range(len(multi_day_itin)):
				self.drawItineraryPath(itin=multi_day_itin[w],colP=next(colorP))
				self.draw_all_attractions(itin=multi_day_itin[w],with_labels=labelsOnOff)
				self.zoomToFit(itin=temp_itin,filename="all_days_"+fname)
		else:	# draws a map for each day
			for w in range(len(multi_day_itin)):
				# draws the city map
				self.drawStreetNetwork()
				self.drawItineraryPath(itin=multi_day_itin[w])
				self.draw_all_attractions(itin=multi_day_itin[w],with_labels=labelsOnOff)
				if fname:
					fname_temp = 'day' + str(w+1) + '_' + fname
					self.zoomToFit(itin=multi_day_itin[w],filename=fname_temp)
				else:
					self.zoomToFit(itin=multi_day_itin[w])
				self.DSmap.clear()


	# formats map
	def zoomToFit(self,itin,filename=False):
		# gets range of longitude and latitude w.r.t. the attractions and zooms map
		lonLatRan = self.getLonLatRange(itin_list=itin)
		margin_range = max([abs(lonLatRan[0][0]-lonLatRan[0][1]), abs(lonLatRan[1][0]-lonLatRan[1][1])])
		margin = 0.5*margin_range
		#self.DSmap.setZoom(-130,25,-60,50)
		#self.DSmap.setZoom(-160,18,-155,23)
		self.DSmap.setZoom(lonLatRan[0][0]-margin, lonLatRan[1][0]-margin, lonLatRan[0][1]+margin, lonLatRan[1][1]+margin)
		ax = self.DSmap.getAxes()
		ax.axes.get_xaxis().set_ticks([])
		ax.get_yaxis().set_ticks([])	
		# fig = matplotlib.pyplot.gcf()
		# fig.set_size_inches(11, 8)	
		if filename: 
			matplotlib.pyplot.savefig(filename,bbox_inches='tight',dpi=120, figsize = (11, 8))
		else: 
			matplotlib.pyplot.show()


	# finds longitude and latitude range w.r.t. lon and lat of attractions
	def getLonLatRange(self,itin_list=False):
		if itin_list: 
			lon_list = [self.attr[self.attr.attraction == h].lon.values[0] for h in itin_list]
			lat_list = [self.attr[self.attr.attraction == h].lat.values[0] for h in itin_list]
			return [[min(lon_list),max(lon_list)],[min(lat_list),max(lat_list)]]
		else: 
			return [[min(self.attr.lon.values),max(self.attr.lon.values)],[min(self.attr.lat.values),max(self.attr.lat.values)]]


	# finds the closest node to the set of addresses
	def findClosestNode(self,placeLat,placeLon,GPH):
	    pos_nodes_zero = True
	    lonLatRange = self.getLonLatRange()
	    lonRange = lonLatRange[0]
	    latRange = lonLatRange[1]
	    # a little more than 2 seconds longitude and latitude
	    latdf = 0.0005
	    londf = 0.0005
	    ct = 0

	    nodesNM_set = [r.split(',') for r in self.gdcon.nodes_iter()]
	    stLines = [[[float(dw) for dw in p.split()] for p in g] for g in nodesNM_set]
	    listLon = scipy.array([u[0][0] for u in stLines])
	    listLat = scipy.array([u[0][1] for u in stLines])

	    while pos_nodes_zero==True and ((latdf<scipy.absolute(latRange[0]-latRange[1])/2) and (londf<scipy.absolute(lonRange[0]-lonRange[1]))):
	        pos_nodes = scipy.array([])
	        nodesNM_set = [r.split(',') for r in self.gdcon.nodes_iter()]
	        ct = ct + 1
	        latidx1 = scipy.where((listLat<placeLat+ct*latdf))
	        tempLatList = listLat[latidx1]
	        tempLonList = listLon[latidx1]
	        nodesNM_set = [nodesNM_set[b] for b in latidx1[0]]
	        latidx = scipy.where(tempLatList>placeLat-ct*latdf)
	        tempLatList = tempLatList[latidx]
	        tempLonList = tempLonList[latidx]
	        nodesNM_set = [nodesNM_set[c] for c in latidx[0]]
	        lonidx1 = scipy.where(tempLonList>placeLon-ct*londf)
	        tempLatList = tempLatList[lonidx1]
	        tempLonList = tempLonList[lonidx1]
	        nodesNM_set = [nodesNM_set[d] for d in lonidx1[0]]
	        lonidx = scipy.where(tempLonList<placeLon+ct*londf)
	        pos_nodes = scipy.array([tempLonList[lonidx],tempLatList[lonidx]])
	        nodesNM_set = [nodesNM_set[e] for e in lonidx[0]]
	        if len(pos_nodes[0])!=0:
	        	pos_nodes_zero = False
	    
	    if len(pos_nodes)==0:
	        return None
	    else:
	        dist = scipy.sqrt(scipy.sum(scipy.array([pos_nodes[0]-placeLon, pos_nodes[1]-placeLat])**2,axis=0))
	        # recall, [lon, lat]
	        idxCN = scipy.where(dist==min(dist))[0][0]
	        closeNode = [pos_nodes[0][idxCN], pos_nodes[1][idxCN]]	    
	        # name of the closest node
	        closeNodeNM = nodesNM_set[idxCN][0]
	        return [closeNodeNM, closeNode]
	

	# finds the total reward given an itinerary
	def getItineraryReward(self,itin):
		# calculates reward for visiting all the attractions in the itinerary
		# reward is in terms of USD
		attr_reward_temp = [self.attr[(self.attr.attraction == nm)].rating.values[0]*10. for nm in itin if (nm != 'hotel')]
		attr_reward = scipy.sum(attr_reward_temp)

		time_reward_temp = [networkx.shortest_path_length(self.gdcon, source=self.findClosestNode(placeLon=self.attr[self.attr.attraction==itin[p]].lon.values[0],placeLat=self.attr[self.attr.attraction==itin[p]].lat.values[0],GPH=self.gdcon)[0], target=self.findClosestNode(placeLon=self.attr[self.attr.attraction==itin[p+1]].lon.values[0],placeLat=self.attr[self.attr.attraction==itin[p+1]].lat.values[0],GPH=self.gdcon)[0], weight='time') for p in range(len(itin)-1)]
		time_reward = scipy.sum(time_reward_temp)*TIME_VALUE_OF_MONEY*(-1)

		return attr_reward + time_reward

	# calculates total travel time
	def getTotalTravelTime(self,itin):
		time_reward_temp = [networkx.shortest_path_length(self.gdcon, source=self.findClosestNode(placeLon=self.attr[self.attr.attraction==itin[p]].lon.values[0],placeLat=self.attr[self.attr.attraction==itin[p]].lat.values[0],GPH=self.gdcon)[0], target=self.findClosestNode(placeLon=self.attr[self.attr.attraction==itin[p+1]].lon.values[0],placeLat=self.attr[self.attr.attraction==itin[p+1]].lat.values[0],GPH=self.gdcon)[0], weight='time') for p in range(len(itin)-1)]
		return scipy.sum(time_reward_temp)


	# calculates total travel time and time spent at all attractions (does not include travel time back to hotel)
	def getTotalTime(self,itin):
		time_reward_temp = [networkx.shortest_path_length(self.gdcon, source=self.findClosestNode(placeLon=self.attr[self.attr.attraction==itin[p]].lon.values[0],placeLat=self.attr[self.attr.attraction==itin[p]].lat.values[0],GPH=self.gdcon)[0], target=self.findClosestNode(placeLon=self.attr[self.attr.attraction==itin[p+1]].lon.values[0],placeLat=self.attr[self.attr.attraction==itin[p+1]].lat.values[0],GPH=self.gdcon)[0], weight='time') for p in range(len(itin)-1)]
		return scipy.sum(time_reward_temp) + 2*(len(itin)-1)


	# solves for the optimal itinerary for one day
	# output: a list of the names of the attractions for one day starting and ending at the hotel
	def solve_optimal_itinerary(self,itin):
		print 'Solving for Optimal Itinerary...'
		print
		# Select n random attractions later in the list to node in question
		maxCost = self.getItineraryReward(itin=itin)
		itinTemp = list(itin)
		# counter for item in itinerary
		i = 1
		totTime = 0
		while totTime<HOURS_AWAY_FROM_HOTEL and i < len(itin):
			switchIdx = i
			rand_attr = scipy.random.randint(low=i, high=len(itin)-1, size=SAMPLE_SIZE)
			for j in rand_attr:
				itinTemp[i], itinTemp[j] = itinTemp[j], itinTemp[i]
				tempCost = self.getItineraryReward(itin=itinTemp)
				if tempCost > maxCost:
					maxCost = tempCost
					switchIdx = j
				itinTemp = list(itin)
			itin[i], itin[switchIdx] = itin[switchIdx], itin[i]
			totTime = self.getTotalTime(itin=itin[0:i])
			i = i + 1			

		itinFinal = itin[0:i-2]
		itinFinal.extend(['hotel'])
		return itinFinal


	# solves itinerary for more than one day, less than or equal to 3 days
	def solve_full_itinerary(self,working_itin,days=1):
		print 'Solving for Full Itinerary...'
		print
		full_itin = []
		# maximum number of days for vacation to allow safe execution of optimal itinerary with max days
		max_days = scipy.floor((len(self.attr.attraction.values)-1)/5.)
		if days > max_days:
			print 'Not enough top attractions for ' + str(days) + ' days of vacation.'
			print 'Maximum days: ' + str(int(max_days))
			print
			days = int(max_days)
		for r in range(days):
			print 'Solving Itinerary for Day ' + str(r+1) + '...'
			print
			# solves optimal itinerary for that day
			day_itin = self.solve_optimal_itinerary(itin=working_itin)
			full_itin.append(day_itin)
			# removes attractions that have been visited in previous days
			for b in day_itin[1:-1]:
				if b in working_itin: 
					working_itin.remove(b)
		return full_itin


if __name__ == '__main__':
	city_itinerary = vacation_itinerary(city_file='Honolulu_edges.csv',attractions_file='Honolulu_nodes.csv')
	#city_itinerary = vacation_itinerary(city_file='austin_edges.csv',attractions_file='austin_nodes.csv')
	#city_itinerary = vacation_itinerary(city_file='District_of_Columbia_edges.csv',attractions_file='District_of_Columbia_nodes.csv')
	#city_itinerary = vacation_itinerary(city_file='Anchorage_edges.csv',attractions_file='Anchorage_nodes.csv')

	# Solve for one optimal path
	#optimalItin = city_itinerary.solve_optimal_itinerary(itin=city_itinerary.initial_itinerary)
	optimalItin = ['hotel', 'USS Arizona Memorial', 'Pacific Aviation Museum Pearl Harbor', 'Koko Crater Trail', 'Bishop Museum', 'hotel']
	#optimalItin = ['hotel', 'Zilker Botanical Garden', 'Lady Bird Johnson Wildflower Center', 'Congress Avenue Bridge / Austin Bats', 'State Capitol', 'hotel']
	#optimalItin = ['hotel', 'Hillwood Museum & Gardens', 'Vietnam Veterans Memorial', 'John F. Kennedy Center for the Performing Arts', 'National Museum of American History', 'hotel']
	#optimalItin = ['hotel', 'Iditarod Trail Sled Dog Race', 'Anchorage Museum at Rasmuson Center', 'hotel']
	print optimalItin
	print 'total reward: ', city_itinerary.getItineraryReward(itin=optimalItin)
	print 'total time: ', city_itinerary.getTotalTime(itin=optimalItin[0:-1])
	print 
	city_itinerary.draw_optimal_itinerary(opt_itin=optimalItin,fname='coast_none_honolulu_.png',labelsOnOff=True,state_lines=False)
	
	"""
	# draws the optimal path without draw_optimal_itinerary
	# city_itinerary.drawItineraryPath(itin=optimalItin)
	# city_itinerary.draw_all_attractions(itin=optimalItin)
	#city_itinerary.zoomToFit(filename='optimal_itinerary.png',itin=optimalItin)
	"""
	"""
	# Solve for multiple day itinerary
	three_day_itin = city_itinerary.solve_full_itinerary(working_itin=city_itinerary.initial_itinerary,days=3)
	city_itinerary.draw_multi_day_optimal_itinerary(multi_day_itin=three_day_itin,fname='three_day_itinerary_new.png',onemap=False,labelsOnOff=True)
	"""
	"""
	# Draw street network and all attraction addresses
	city_itinerary.drawStreetNetwork()
	city_itinerary.draw_all_attractions(itin=city_itinerary.initial_itinerary,with_labels=True)
	city_itinerary.zoomToFit(itin=city_itinerary.initial_itinerary,filename='test.png')
	"""
	
	
	