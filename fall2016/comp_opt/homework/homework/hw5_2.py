import pandas
import networkx
import pyomo
import pyomo.environ as pe
import geoplotter
import re
import scipy
import matplotlib.pyplot as plt

class roadNetwork:
	"""Class for taking input street data (as a .csv) and creating
	a network structure for the city.

	edges: streets, where two-way streets act as two edges; one-way streets
		specified by ONE_WAY column, where 'FT' has same direction as shape
		and 'TF' has reverse direction. All other values are two-way streets

	nodes: ends of streets, specified by lat/lon nodes"""

	def __init__(self,street_data, address_data):
		"""street_data must be a string representing a .csv file location"""

		# Initialize edge_data
		self.edge_data = pandas.read_csv(street_data,usecols=['ONE_WAY','MILES','kmlgeometry']) #'SEGMENT_ID',
		
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
		self.node_data['lat'] = self.node_data.nodeid.copy().str.split().str[1] # separate lat
		self.node_data['lat'] = self.node_data['lat'].apply(pandas.to_numeric) # convert lat to float
		# Longitude
		self.node_data['lon'] = self.node_data.nodeid.copy().str.split().str[0] # separate lon
		self.node_data['lon'] = self.node_data['lon'].apply(pandas.to_numeric) # convert lon to float
		self.node_data = self.node_data.reset_index()

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

		# Set indices for easy indexing
		# self.node_data.set_index(['latlon'],inplace=True)
		# self.edge_data.set_index(['source','target'],inplace=True)

		# Initialize addresses
		self.addresses = pandas.read_csv(address_data)
		self.addresses['name'] = self.addresses.Address.str.extract(
			'([a-zA-Z -]+),',expand=True)
		# self.num_edges = len(self.edge_data)

		# Initialize geoplotter instance
		self.map = geoplotter.GeoPlotter()

	def create_graph(self):
		# Add edges
		self.graph = networkx.DiGraph()
		edges = zip(list(self.edge_data.source.values), 
			list(self.edge_data.target.values), list(self.edge_data.MILES.values))
		self.graph.add_weighted_edges_from(edges)

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

	def latlon_from_address(self,name):
		tempdf = self.addresses.loc[self.addresses.name == name]
		return (tempdf.Lat.values, tempdf.Lon.values)

	def draw_edges(self,lw=0.3):
		data = self.edge_data['edge'].str.split(',')
		edges = [[[float(e) for e in val.split()] for val in ll] for ll in data]
		self.map.drawLines(lines=edges,color='b',linewidth=lw)

	def draw_nodes(self,currentloc,sz=50):
		# Draw current location in green
		lat,lon = self.latlon_from_address(currentloc)
		self.map.drawPoints(lat=lat,lon=lon,color='g',s=sz)
		# Draw destinations in red
		destlat = self.addresses[self.addresses.name != currentloc]['Lat'].values
		destlon = self.addresses[self.addresses.name != currentloc]['Lon'].values
		self.map.drawPoints(lat=destlat,lon=destlon,color='r',s=sz)


	def get_shortestpath_nx(self,start,end):
		""" Return shortest path in self.graph between 'start' and 'end' nodes"""
		try:
			self.graph # check for self.graph
		except AttributeError:
			self.create_graph() # create self.graph
		return networkx.shortest_path(self.graph,source=start,target=end,weight='weight')

	def get_shortestpath_pyomo(self,start,end):
		try:
			self.graph # check for self.graph
		except AttributeError:
			self.create_graph() # create self.graph

		self.m = pe.ConcreteModel()

# Create edges set from pyomo only
		# self.edge_data.set_index(['source','target'],inplace=True)
		# self.edge_data.sort_index(inplace=True)
		# self.edge_set = self.edge_data.index.unique()
		# self.m.edge_set = pe.Set(initialize=self.edge_set, dimen=2)

		# Create nodes set
		self.m.node_set = pe.Set(initialize=self.node_data.nodeid.values)
		# # Create edges set from networkx graph
		self.m.edge_set = pe.Set(initialize=self.graph.edges(),ordered=True)

		# Create variables
		self.m.Y = pe.Var(self.m.edge_set, domain=pe.NonNegativeReals)

		# Create objective
		def obj_rule(m):
			# Set up objective with edges populated with MILES (length) data
			# return sum(self.m.Y[e] * self.edge_data.ix[e,'MILES'].values for e in self.edge_set)
			return sum(m.Y[(u,v)] * self.graph[u][v]['weight'] for u,v,d \
				in self.graph.edges_iter(data=True))
		self.m.OBJ = pe.Objective(rule=obj_rule, sense=pe.minimize)

		def flow_bal_rule(m, n):
			# edges = self.edge_data.reset_index()
			# preds = edges[ edges.target == n ]['source']
			# succs = edges[ edges.source == n ]['target']
			# return sum(m.Y[(p,n)] for p in preds) - sum(m.Y[(n,s)] for s in succs) \
			# 	== self.node_data.ix[n,'Imbalance']
			if n == start:
				imb = -1 # imbalance
			elif n == end:
				imb = 1 # imbalance
			else:
				imb = 0 # imbalance

			preds = self.graph.predecessors(n)
			succs = self.graph.successors(n)
			return sum(m.Y[(p,n)] for p in preds) - sum(m.Y[(n,s)] for s in succs) == imb
		self.m.FlowBal = pe.Constraint(self.m.node_set, rule=flow_bal_rule)

		# Solve the model
		solver = pyomo.opt.SolverFactory('gurobi') # 'cplex'
		results = solver.solve(self.m, tee=True, keepfiles=False, 
			options_string='mip_tolerances_integrality=1e-9 mip_tolerances_mipgap=0')

		if (results.solver.status != pyomo.opt.SolverStatus.ok):
			logging.warning('Check solver not ok?')
		if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):  
			logging.warning('Check solver optimality?')

		# Collect results for mapping
		ans = [start]
		c = start

		while c != end: # while current is not the last node
			succs = self.graph.successors(c)
			nxt = [s for s in succs if self.m.Y[(c,s)] == 1] # get list of successors
			ans.append(nxt[0]) # take first successor
			c = nxt[0]
		return ans # list of str(nodes) in shortest path 

		self.edge_data.reset_index() # reset to avoid conflicts with other functions

	def draw_route(self,route,c='y',lw=5):
		edges = [[[float(e) for e in val.split()] for val in route]]
		self.map.drawLines(lines=edges,color=c,linewidth=lw)

if __name__ == '__main__':
	street_data = 'hw05_files/austin.csv'
	address_data = 'hw05_files/addresses.csv'
	nw = roadNetwork(street_data,address_data)

	currentloc = 'Engineering Teaching Center'
	start = nw.get_closest_node(currentloc)

	# Draw map
	nw.draw_edges()
	nw.draw_nodes(currentloc=currentloc)
	nw.map.setZoom(-97.8526, 30.2147, -97.6264, 30.4323)
	# plt.show()
	# plt.savefig('network.png')

	# Get shortest paths to Hula Hut
	end = nw.get_closest_node('Hula Hut')
	nxpath = nw.get_shortestpath_nx(start,end) # from networkx
	pyomopath = nw.get_shortestpath_pyomo(start,end) # from pyomo
	# Draw shortest paths to Hula Hut
	nw.map.clear()
	nw.draw_route(nxpath,c='green',lw=6) # green
	nw.draw_route(pyomopath,c='orange',lw=3) # orange
	nw.draw_edges()
	nw.draw_nodes(currentloc)
	nw.map.setZoom(-97.796, 30.278, -97.725, 30.314)
	# plt.show()
	# plt.savefig('hulahut.png')

	# Get shortest path to Rudys Country Store and Bar-B-Q
	end = nw.get_closest_node('Rudys Country Store and Bar-B-Q')
	nxpath = nw.get_shortestpath_nx(start,end) # from networkx
	pyomopath = nw.get_shortestpath_pyomo(start,end) # from pyomo
	# Draw shortest paths to Rudys Country Store and Bar-B-Q
	nw.map.clear()
	nw.draw_route(nxpath,c='green',lw=6) # green
	nw.draw_route(pyomopath,c='orange',lw=3) # orange
	nw.draw_edges()
	nw.draw_nodes(currentloc)
	nw.map.setZoom(-97.8526, 30.2147, -97.6264, 30.4323)
	plt.show()
	# plt.savefig('rudys.png')

	# Zoom in to Rudy's U-Turn
	nw.map.clear()
	nw.draw_route(nxpath,c='green',lw=6) # green
	nw.draw_route(pyomopath,c='orange',lw=3) # orange
	nw.draw_edges()
	nw.draw_nodes(currentloc)
	nw.map.setZoom(-97.7654, 30.375, -97.7241, 30.4225)
	plt.show()
	# plt.savefig('rudyszoom.png')