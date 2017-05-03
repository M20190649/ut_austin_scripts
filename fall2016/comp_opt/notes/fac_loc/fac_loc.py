import pandas
import networkx
import pyomo
import pyomo.environ as pe


class FacLoc:
    def __init__(self, budget, distances_file, costs_file):
        self.budget = budget
        self.distdf = pandas.read_csv(distances_file)
        self.opencdf = pandas.read_csv(costs_file)

        self.fac_set = self.opencdf.location.unique()
        self.cust_set = list(set(self.distdf.startNode.unique()).union(self.distdf.destNode.unique()))

        # Change the index of opencdf
        self.opencdf.set_index(['location'], inplace=True)

        self.computePairwiseDist()
        self.createModel()

    def computePairwiseDist(self):
        # Convert distdf into a networkx graph
        g = networkx.Graph()
        for idx, data in self.distdf.iterrows():
            g.add_edge(data['startNode'], data['destNode'], length=data['dist'])

        self.distances = networkx.algorithms.all_pairs_dijkstra_path_length(g, weight='length')

    def createModel(self):
        # Create the model object
        m = pe.ConcreteModel()

        # Create the sets
        m.fac_set = pe.Set(initialize=self.fac_set)
        m.cust_set = pe.Set(initialize=self.cust_set)

        # Create the variables
        m.y = pe.Var( m.fac_set, domain=pe.Binary)
        m.x = pe.Var( m.cust_set * m.fac_set, domain=pe.Binary)

        # Define objective
        def obj_rule(m):
            return sum( m.y[i] * self.opencdf.ix[i,'cost'] for i in m.fac_set) + sum( m.x[j,i] * self.opencdf.ix[j,'ncust'] * self.distances[j][i] for j in m.cust_set for i in m.fac_set)
        m.obj = pe.Objective(rule = obj_rule, sense=pe.minimize)

        def cust_rule(m, cust):
            return sum( m.x[cust, fac] for fac in m.fac_set ) == 1
        m.custConst = pe.Constraint( m.cust_set, rule=cust_rule)

        def budget_rule(m):
            return sum( m.y[fac] for fac in m.fac_set) == self.budget
        m.budgetConst = pe.Constraint( rule = budget_rule)

        def fac_open_rule(m, cust, fac):
            return m.x[cust, fac] <= m.y[fac]
        m.facOpenConst = pe.Constraint( m.cust_set*m.fac_set, rule=fac_open_rule)
        
        self.m = m

    def solve(self):
        solver = pyomo.opt.SolverFactory('cplex')
        results = solver.solve(self.m, tee=True, keepfiles=False, options_string="mip_tolerances_integrality=1e-9 mip_tolerances_mipgap=0")

if __name__ == '__main__':
    fc = FacLoc(1, 'distances.csv','nodeData.csv')
    fc.solve()
