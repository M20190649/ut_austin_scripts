import pandas
import pyomo
import pyomo.environ as pe
import pyomo.opt
import logging
import scipy

node_data = pandas.read_csv('nodes.csv')
arc_data = pandas.read_csv('arcs.csv')

M = 1000

node_data.set_index(['node'],inplace=True)
node_set = sorted(node_data.index.unique())

# Get predecesors and successors
preds = {}
succs = {}
for n in node_set:
	preds[n] = arc_data[ arc_data.endNode == n ].startNode
	succs[n] = arc_data[ arc_data.startNode == n ].endNode

arc_data.set_index(['startNode','endNode'],inplace=True)
arc_set = sorted(arc_data.index.unique())

# Create model
m = pe.ConcreteModel() 

# Create sets assuming list
m.node_set = pe.Set(initialize=node_set) # ie. nodes
m.arc_set = pe.Set(initialize=arc_set,dimen=2) # ie. arcs
m.res_set = pe.Set(initialize=['time','volume']) # ie. resources

# Create variables
m.X = pe.Var(m.arc_set,domain=pe.Binary)

def t_bounds(m,r,n):
	return ( node_data.ix[n,r+'LB'], node_data.ix[n,r+'UB'] )
m.T = pe.Var(m.res_set*m.node_set,bounds=t_bounds) # x-prod; T for every (resource,node) pair

# Create objective
def obj_rule(m):
	return sum( arc_data.ix[e,'cost'] * m.X[e] for e in m.arc_set )
m.OBJ = pe.Objective(rule=obj_rule,sense=pe.minimize)

# Create contraints
def flow_rule(m,u):
	rhs = 0
	if u == 'start':
		rhs = -1
	elif u == 'end':
		rhs = 1
	return sum( m.X[a,u] for a in preds[u] ) - sum( m.X[u,a] for a in succs[u] ) == rhs
m.flowConst = pe.Constraint(m.node_set,rule=flow_rule) # create constraint for all nodes

def res_rule(m,r,u,v):
	return m.T[r,u] + arc_data.ix[(u,v),r] - m.T[r,v] + M * m.X[u,v] <= M
m.resConst = pe.Constraint(m.res_set*m.arc_set,rule=res_rule)

# Solve the model
solver = pyomo.opt.SolverFactory('gurobi')
results = solver.solve(m, tee=True, keepfiles=False, options_string="mip_tolerances_integrality=1e-9 mip_tolerances_mipgap=0")

if (results.solver.status != pyomo.opt.SolverStatus.ok):
    logging.warning('Check solver not ok?')
if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):  
    logging.warning('Check solver optimality?') 

# Print results
cur = 'start'
path = [cur]
while cur != 'end':
	for n in succs[cur]:
		if m.X[cur,n].value == 1: cur = n
	path.append(cur)

print path

for i,j in zip(scipy.arange(len(path)),scipy.arange(1,len(path)+1)):
	if j != len(path):
		print 'Edge (' + path[i] + ', ' + path[j] + ') in the shortest path'