import pandas
import scipy
import geoplotter
import networkx

class ESPRC:

	def __init__(self,street_data, address_data):
		"""street_data must be a string representing a .csv file location"""

		# Initialize edge_data
		self.edge_data = pandas.read_csv(street_data,usecols=['ONE_WAY','MILES','kmlgeometry'])

		# Initialize addresses
		self.addresses = pandas.read_csv(address_data)

		# Initialize geoplotter instance
		self.map = geoplotter.GeoPlotter()

		# Create city and address networkx graphs
		self.create_edge_graph()
		self.create_addr_graph()

	def create_edge_graph(self):
		# Create edge graph
		self.edge_graph = networkx.DiGraph()

		# Extract start- and end-nodes from edge pairs, and return as df (expand=True)
		start_nodes = self.edge_data.kmlgeometry.str.extract(
			'LINESTRING \(([0-9-.]* [0-9-.]*),',expand=True)
		end_nodes = self.edge_data.kmlgeometry.str.extract(
			'([0-9-.]* [0-9-.]*)\)',expand=True)
		self.edge_data.drop('kmlgeometry', axis=1, inplace=True)

		# Combine latlon data and drop duplicates
		self.node_data = pandas.concat([start_nodes,end_nodes]).drop_duplicates()
		
		# Create two columns, 'nodeid' (string) and 'latlon' (float)
		self.node_data.columns = ['nodeid'] # label string latlon as nodeid
		
		# Latitude
		self.node_data['lat'] = \
			self.node_data.nodeid.copy().str.split().str[1] # separate lat
		self.node_data['lat'] = \
			self.node_data['lat'].apply(pandas.to_numeric) # convert lat to float
		
		# Longitude
		self.node_data['lon'] = \
			self.node_data.nodeid.copy().str.split().str[0] # separate lon
		self.node_data['lon'] = \
			self.node_data['lon'].apply(pandas.to_numeric) # convert lon to float
		
		self.node_data = self.node_data.reset_index() # reset to avoid possible conflicts

		# Assign local variables for lat/lon arrays
		self.node_lats = scipy.array(self.node_data.lat.values)
		self.node_lons = scipy.array(self.node_data.lon.values)

		# Add 'source' and 'target' columns to edge_data dataframe
		self.edge_data['source'] = start_nodes
		self.edge_data['target'] = end_nodes

		# Create reversed dataset for where ONE_WAY = TF or B
		self.edge_data_reverse = self.edge_data.copy()
		self.edge_data_reverse.loc[:,['source','target']] = \
			self.edge_data_reverse.loc[:,['target','source']].values

		# Reverse source/target where ONE_WAY = TF
		TF_idx = self.edge_data['ONE_WAY'] == 'TF'
		self.edge_data.loc[TF_idx, ['source','target']] = \
			self.edge_data.loc[TF_idx, ['target','source']].values

		# Add reverse (and keep original) where ONE_WAY = B
		B_idx = self.edge_data['ONE_WAY'] == 'B'
		rev_b = self.edge_data.loc[B_idx, ['source','target','MILES']]
		rev_b.loc[:,['source','target']] = rev_b.loc[:,['target','source']].values
		self.edge_data = pandas.concat([self.edge_data,rev_b]) # add to edge_data df
		self.edge_data.drop('ONE_WAY', axis=1, inplace=True) # drop ONE_WAY column
		self.edge_data = self.edge_data.reset_index()
		self.edge_data.drop('index', axis=1, inplace=True) # drop index column
		
		self.edge_data['edge'] = self.edge_data.source + ',' + \
			self.edge_data.target

		# Add edge set to graph
		edges = zip(list(self.edge_data.source.values), 
			list(self.edge_data.target.values), list(self.edge_data.MILES.values))
		self.edge_graph.add_weighted_edges_from(edges)

	def create_addr_graph(self):

		# Create node graph
		self.addr_graph = networkx.DiGraph()

		# Find all closest nodes
		self.addresses['closest'] = [self.get_closest_node(s)\
			for s in self.addresses.ShortHand.values]

		#  artificial end-node identical to start-node
		newline = self.addresses.tail(1)
		self.addresses = pandas.concat([self.addresses,newline],
			ignore_index=True)
		self.addresses.loc[15,'ShortHand'] = 'ETC_Start'
		self.addresses.loc[16,'ShortHand'] = 'ETC_End'

		# Create edge set
		# Get all start,end pairs
		address_edges = [(s,e) for s in self.addresses.ShortHand.values\
			for e in self.addresses.ShortHand.values if (s!=e) and (s!='ETC_End') and e!='ETC_Start']
		start_edges,end_edges = zip(*address_edges)

		# Define shortest paths and times for all start,end pairs
		paths = []
		times = []
		for s,e in zip(start_edges,end_edges):
			paths.append(self.get_shortestpath_nx(
				self.addresses[self.addresses.ShortHand==s].closest.values[0],
				self.addresses[self.addresses.ShortHand==e].closest.values[0]))
			times.append(networkx.shortest_path_length(G=self.edge_graph,
			source=self.addresses[self.addresses.ShortHand==s].closest.values[0],
			target=self.addresses[self.addresses.ShortHand==e].closest.values[0],
			weight='time'))

		# Retrieve descriptive data for all edges
		loadtimes = []
		values = []
		volumes = []
		timestarts = []
		timeends = []
		for e in end_edges:
			loadtimes.append(self.addresses[self.addresses.ShortHand==e].LoadingTime.values[0])
			values.append(self.addresses[self.addresses.ShortHand==e].Value.values[0])
			volumes.append(self.addresses[self.addresses.ShortHand==e].Volume.values[0])
			timestarts.append(self.addresses[self.addresses.ShortHand==e].TimeWindowStart.values[0])
			timeends.append(self.addresses[self.addresses.ShortHand==e].TimeWindowEnd.values[0])

		# Compile descriptive characteristics into a list of dictionaries
		edge_chars = [dict(path=paths[i],time=times[i],loadtime=loadtimes[i],
			value=values[i],volume=volumes[i],timestart=timestarts[i],timeend=timeends[i])\
			for i in range(len(times))]
		final_edges = zip(start_edges,end_edges,edge_chars)
		# Add edge set to graph
		self.addr_graph.add_edges_from(final_edges)

	def get_closest_node(self,name):
		"""Determine closest network node to supplied latlon pair"""
		Re = 3959 # Approximate radius of the earth [miles]
		lat,lon = self.latlon_from_address(name)
		dlat = Re * abs(self.node_lats - lat) * scipy.pi/180 # [miles]
		dlon = Re * abs(self.node_lons - lon) * scipy.pi/180 * \
			scipy.cos(lat*scipy.pi/180) # [miles]
		dlatlon = dlat + dlon
		closest_idx = scipy.argmin(dlatlon)
		return self.node_data.loc[closest_idx,['nodeid']].values[0] # str(latlon)

	def latlon_from_address(self,shrthnd):
		tempdf = self.addresses.loc[self.addresses.ShortHand == shrthnd]
		return (tempdf.Lat.values, tempdf.Lon.values)

	def get_shortestpath_nx(self,start,end):
		""" Return shortest path in self.graph between 'start' 
		and 'end' nodes using networkx"""
		return networkx.shortest_path(self.edge_graph,source=start,target=end,weight='time')

	def draw_edges(self,lw=0.3):
		data = self.edge_data['edge'].str.split(',')
		edges = [[[float(e) for e in val.split()] \
			for val in ll] for ll in data] # edges as list of lists
		self.map.drawLines(lines=edges,color='b',linewidth=lw)

	def draw_nodes(self,locs,sz=50):
		# Draw destinations in red
		destlats = []
		destlons = []
		labels = locs
		for l in locs:
			destlat.append(self.addresses[self.addresses.ShortHand==l]['Lat'].values[0])
			destlon.append(self.addresses[self.addresses.ShortHand==l]['Lon'].values[0])
		self.map.drawPoints(lat=destlats,lon=destlons,color='r',s=sz)

		# Label attractions
		labels = [str(self.addresses[self.addresses.ShortHand==l].index.values[0])\
			for l in labels]
		for l,x,y in zip(labels,destlats,destlons):
			self.map.annotate(text=l,xy=(x,y),size=10)

	def draw_route(self,route,c='y',lw=5):
		edges = [[[float(e) for e in val.split()] for val in route]] # edges as list of lists
		self.map.drawLines(lines=edges,color=c,linewidth=lw)

	def solve(self,start_time,end_time,volume,fname=False):
		
		# Start from ETC_start
		soln = set()
		succs = list(self.addr_graph.successors('ETC_Start'))
		succs.remove('ETC_End')

		# Create blank list for holding labels
		networkx.set_node_attributes(G=self.addr_graph,name='labels',values=[])

		# Solve
		for i in range(len(succs)):
			start_edge = self.addr_graph['ETC_Start'][succs[i]]
			arr_time = [start_time]
			dep_time = [end_time]
			arr_vol = [0,0]
			
			# Arrival time
			if start_time < min(self.addresses.TimeWindowStart.values):
				arr_time.append(start_edge['timestart'])
			else: 
				arr_time.append(start_time + start_edge['time']/3600.)
			
			# Departure time
			dep_time.append(arr_time[-1] + start_edge['loadtime'])

			# Other parameters
			if (arr_time[-1] >= start_edge['timestart']) and (dep_time[-1] <= start_edge['timeend']):
				vol_left = volume - start_edge['volume']
				reward = start_edge['value']
				path = ['ETC_Start']
				path.append(succs[i])
				unreach_nodes = set()
				unreach_nodes.add(succs[i])
				extend = True
				arr_vol.append(start_edge['volume'])

				# Create dictionary of information to this successor
				node_dict = [dict(zip(
					('arr_time','dep_time','vol_left','reward','path','unreach_nodes','extend','arr_vol'),
					(arr_time,dep_time,vol_left,reward,path,unreach_nodes,extend,arr_vol)))]
				
				# Add data to 'labels' information list in networkx graph
				self.addr_graph.node[str(succs[i])]['labels'] = node_dict
				soln.add(succs[i])

		r = 1
		end = False
		while end == False:
			cur_node = list(soln)[0]
			all_succs = self.addr_graph.successors(cur_node)
			labels_to_extend = [l for l in list(self.addr_graph.node[str(cur_node)]['labels']) if l['extend'] == True]

			for j in range(len(labels_to_extend)):
				cur_node_labels = labels_to_extend[j]
				succs = [s for s in all_succs if (\
					not (s in cur_node_labels['unreach_nodes'])\
					and (cur_node_labels['dep_time'][-1] + self.addr_graph[cur_node][s]['time']/3600. <= self.addr_graph[cur_node][s]['timeend'])\
					and (cur_node_labels['dep_time'][-1] + self.addr_graph[cur_node][s]['time']/3600. + self.addr_graph[cur_node][s]['loadtime'] <= end_time)\
					and (cur_node_labels['dep_time'][-1] + self.addr_graph[cur_node][s]['time']/3600. >= self.addr_graph[cur_node][s]['timestart'] - 05)\
					and (cur_node_labels['vol_left'] >= self.addr_graph[cur_node][s]['volume']))]

				for k in range(len(succs)):
					cur_succ = succs[k]
					succ_edge = self.addr_graph[cur_node][cur_succ]
					arr_time = list(cur_node_labels['arr_time'])
					arr_time.append(cur_node_labels['dep_time'][-1] + succ_edge['time']/3600.)
					if arr_time[-1] < succ_edge['timestart']: arr_time[-1] = succ_edge['timestart']
					vol_left = cur_node_labels['vol_left'] - succ_edge['volume']

					if cur_succ == 'ETC_End' and vol_left == 0 or cur_node_labels['arr_vol'][-1] >=\
					min(0.75*scipy.sum(self.addresses.Volume.values), 0.5*volume) or cur_succ != 'ETC_End':
						
						dep_time = list(cur_node_labels['dep_time'])
						dep_time.append(arr_time[-1] + succ_edge['loadtime'])

						path = list(cur_node_labels['path'])
						path.append(cur_succ)

						unreach_nodes = set(cur_node_labels['unreach_nodes'])
						unreach_nodes.add(cur_succ)

						reward = cur_node_labels['reward'] + succ_edge['value']

						extend = True

						arr_vol = list(cur_node_labels['arr_vol'])
						if cur_succ != 'ETC_End':
							arr_vol.append(arr_vol[-1] + succ_edge['volume'])

						# Determine dominance
						succ_labels = self.addr_graph.node[str(cur_succ)]['labels']
						add_label = True
						for x in range(len(succ_labels)):
							cur_succ_labels = succ_labels[x]

							if add_label == True: 
								if unreach_nodes <= cur_succ_labels['unreach_nodes'] and\
								reward >= cur_succ_labels['reward'] and\
								dep_time[-1] <= cur_succ_labels['dep_time'][-1] and\
								vol_left >= cur_succ_labels['vol_left'] and\
								(dep_time[-1],vol_left,reward) !=\
								(cur_succ_labels['dep_time'][-1], cur_succ_labels['vol_left'], cur_succ_labels['reward']): 
									node_dict = list(self.addr_graph.node[str(cur_succ)]['labels'])
									node_dict.remove(cur_succ_labels)
									self.addr_graph.node[str(cur_succ)]['labels'] = node_dict

								elif unreach_nodes >= cur_succ_labels['unreach_nodes'] and\
								abs(dep_time[-1] - start_time) >= abs(cur_succ_labels['dep_time'][-1] - start_time) and\
								reward <= cur_succ_labels['reward'] and\
								vol_left <= cur_succ_labels['vol_left'] and\
								(dep_time[-1],vol_left,reward) !=\
								(cur_succ_labels['dep_time'][-1],cur_succ_labels['vol_left'],cur_succ_labels['reward']):
									add_label = False

							if add_label:
								node_dict = list(self.addr_graph.node[str(cur_succ)]['labels'])
								node_dict.append(dict(zip(
									('arr_time','dep_time','vol_left','reward','path','unreach_nodes','extend','arr_vol'),
									(arr_time,dep_time,vol_left,reward,path,unreach_nodes,extend,arr_vol))))
								self.addr_graph.node[str(cur_succ)]['labels'] = node_dict

								if not (cur_succ in soln) and (cur_succ != 'ETC_End'):
									soln.add(cur_succ)

				cur_node_labels['extend'] = False

			r += 1

			ext = [l['extend'] for l in self.addr_graph.node[str(cur_node)]['labels']]
			if scipy.all(ext) == False:
				soln.remove(cur_node)

			if len(soln) == 0 or r > 250:
				end = True

		if len(self.addr_graph.node['ETC_End']['labels']) == 0: route = []
		else: 
			vol = [v['vol_left'] for v in self.addr_graph.node['ETC_End']['labels']]
			if len(vol) == 0: it = [y for y in self.addr_graph.node['ETC_End']['labels'] if y['dep_time'][-1] == end_time]
			else:
				min_vol = min(vol)
				it = [y for y in self.addr_graph.node['ETC_End']['labels'] if y['vol_left'] == min_vol]
			min_costs = [-c['reward'] for c in it]
			min_cost = min(min_costs)
			opt_cost = [p for p in it if -p['reward'] == min_cost]
			route = opt_cost[0]

			opt_route = pandas.DataFrame(route['path'],columns=['ShortHand'])
			opt_route['ArrivalTime'] = route['arr_time']
			opt_route['DepartureTime'] = route['dep_time']
			opt_route['ArrivalVolume'] = route['arr_vol']

			if fname:
				opt_route.to_csv(fname)

			if draw: 
				route = list(route['path'])
				self.draw_edges()
				for r in range(len(route)-1):
					self.draw_route(self.addr_graph[route[i]][route[i+1]]['path'])
				self.draw_nodes(route)
				self.zoom(-97.8526, 30.2147, -97.6264, 30.4323)
				import matplotlib.pyplot as plt
				plt.savefig(draw)


		print 'Route value: ', str(route['reward'])
		return route

if __name__ == '__main__':

	city = 'hw06_files/austin.csv'
	addresses = 'hw06_files/addresses.csv'
	esprc = ESPRC(city,addresses)

	# Route 1
	route1 = esprc.solve(start_time=9,end_time=12,volume=50)

	# Route 2
	route2 = esprc.solve(start_time=8,end_time=17,volume=100,fname='schedule.csv',draw='schedule.png')

	# Route 3
	route3 = esprc.solve(start_time=7,end_time=18,volume=250)