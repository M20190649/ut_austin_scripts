import pandas
import pyomo
import pyomo.environ as pe
import pyomo.opt

class MILP:

	def __init__(self,dic=False,**kwargs):
		"""Initializes solution to Mixed Integer Linear Program (MILP)
		where path1 and path2 are two input paths to .csv files"""
		
		self.kwargs = kwargs

		def data_init(path,idx):
			data = pandas.read_csv(path)
			data.set_index(idx,inplace=True)
			data.sort_index(inplace=True)
			return data

		if dic:
			import os
			self.data = {}
			self.sets = {}
			for path,idx in kwargs.iteritems(): 
				newkey = os.path.splitext(path)[0] # dict keys are filenames
				self.data[newkey] = data_init(path,idx)
				# Set up based on indices
				self.sets[newkey] = self.data[newkey].index.unique()
			# example: self.data['nodes'] or self.sets['nodes']
		else:
			self.data = []
			self.sets = []
			for i,(path,idx) in enumerate( kwargs.iteritems() ): 
				self.data.append( data_init(path,idx) )
				# Set up based on indices
				self.sets.append( self.data[i].index.unique() )
			# example: self.data[0] or self.sets[0]

		self.create_model(dic=False)

	def create_model(self,dic=False):
		""" Construct pyomo for MILP"""
		self.m = pe.ConcreteModel() # Create model

		# Create sets assuming list
		self.m.sets = []
		for i,s in enumerate(self.sets):
			print s
			if isinstance( s[0], tuple):
				newset = pe.Set(initialize=s, dimen=len(s[0]))
				self.m.sets.append( newset )
			else:
				newset = pe.Set(initialize=s)
				self.m.sets.append( newset )

		for path,idx in kwargs.iteritems(): 
			newkey = os.path.splitext(path)[0] # dict keys are filenames

		for i in self.m.sets:
			print i




if __name__ == '__main__': 

	items = {}

	paths = ['nodes.csv','arcs.csv']
	indices = [['node'],['startNode','endNode']]

	for p,i in zip(paths,indices):
		items[p] = i

	milp = MILP(dic=False,**items)
	# milp = MILP(path1=path1,path2=path2,idx1=['node'],idx2=['startNode','endNode'])