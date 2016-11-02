import pandas
import networkx
import pyomo
import pyomo.environ as pe

class roadNetwork:
	"""Class for taking input street data (as a .csv) and creating
	a network structure for the city.

	edges: streets, where two-way streets act as two edges; one-way streets
		specified by ONE_WAY column, where 'FT' has same direction as shape
		and 'TF' has reverse direction. All other values are two-way streets

	nodes: ends of streets, specified by lat/lon nodes"""

	def __init__(self,street_data, address_data):
		"""street_data must be a string representing a .csv file location"""

		#initialize streets
		self.streets = pandas.read_csv(street_data,usecols=['SEGMENT_ID','ONE_WAY','MILES','kmlgeometry'])
		start_nodes = self.streets.kmlgeometry.str.extract('LINESTRING \(([0-9-.]* [0-9-.]*),',expand=False)
		self.streets['start_nodes'] = start_nodes # first latlon pair
		end_nodes = self.streets.kmlgeometry.str.extract('([0-9-.]* [0-9-.]*)\)',expand=False)
		self.streets['end_nodes'] = end_nodes # last latlon pair
		# initialize addresses
		self.addresses = pandas.read_csv(address_data)
		self.numstreets = len(self.streets)

	def create_network(self):

		roadnw = networkx.DiGraph()
		for i in range(self.numstreets):
			start = self.streets['start_nodes'][i]
			end = self.streets['end_nodes'][i]
			roadnw.add_edge( start, end )


		# print roadnw.edges()

	def get_closest_node(self,llpair):
		"""Determine closest network node to supplied latlon pair"""

		pass




if __name__ == '__main__':
	street_data = 'hw05_files/austin.csv'
	address_data = 'hw05_files/addresses.csv'
	nw = roadNetwork(street_data,address_data)
	nw.create_network()
