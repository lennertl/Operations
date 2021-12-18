#########################################################################################
#################################### PACKAGES ###########################################
#########################################################################################

import numpy as np
import os
import pandas as pd
import time
from gurobipy import Model,GRB,LinExpr
import pickle
from copy import deepcopy

# get distances df
os.chdir("../data")
cwd = os.getcwd()

#########################################################################################
################################### MODEL SETUP #########################################
#########################################################################################

# get the instance name that is corresponding to the data
instance_name = 'grid.csv'

distance_name = 'distances.csv'
daytime_name = 'day_timematrix.csv'
eveningtime_name = 'evening_timematrix.csv'
nighttime_name = 'night_timematrix.csv'

distance_frame = pd.read_csv(os.path.join(cwd,distance_name))
daytime_frame = pd.read_csv(os.path.join(cwd,daytime_name))
eveningtime_frame = pd.read_csv(os.path.join(cwd,eveningtime_name))
nighttime_frame = pd.read_csv(os.path.join(cwd,nighttime_name))

# load the data for this instance
customers = pd.read_csv(os.path.join(cwd,instance_name))

# Keeping track of the start time of the model (for overall performance eval)
startTime = time.time()

# Initializing an empty Gurobi model
model = Model()

#########################################################################################
#################################### VARIABLES ##########################################
#########################################################################################

# Binary decision variable for every possible route, first consider case for only 1 vehicle, N = 100
x = {}	# 100x100
for i in range(len(customers)):
	for j in range(len(customers)):
		x[i,j] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY,name="x[%s,%s]"%(i,j))

# List of linehaul customers
lh = np.where(customers.iloc[:,3].to_numpy() == False)[0]
# List of backhaul customers
bh = np.where(customers.iloc[:,3].to_numpy() == True)[0]

# Binary decision variable =1 if customer is served
# Linhaul customers
u = {}
u[0] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name="u[0]")		# depot needs to be available
for i in lh:
	u[i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name="u[%s]"%(i))

# Backhaul customers
v = {}
v[0] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name="v[0]")		# depot needs to be available
for i in bh:
	v[i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name="v[%s]"%(i))	
    
model.update()

#########################################################################################
################################# OBJECTIVE #############################################
#########################################################################################

edges = {}
edges['Distance'] = []
edges['From'] = []
edges['To'] = []
for i in range(100):
	for j in range(100):
		edges['Distance'].append(distance_frame[str(i)][j])
		edges['From'].append(i)
		edges['To'].append(j)

obj = LinExpr()

c_r = 0.01
alpha = c_r  # assuming flat surface
beta = 3  # drag
w = 10000  # truck weight
f = 0  # load carried
vel = 90 / 3.6  # velocity used for drag calculation

for i in range(0,len(edges)):
	obj += (alpha*(w+f)+beta*vel**2)*edges['Distance'][i]*x[edges['From'][i],edges['To'][i]]


model.setObjective(obj, GRB.MINIMIZE)
model.update()

#########################################################################################
################################### CONSTRAINTS #########################################
#########################################################################################

depot = 43		# after doing np.sqrt((x-x0)**2+(y-y0)**2) for all nodes

# Equation 7
thisLHS = LinExpr()
a = 1000		# [kg], amount of product the lh customer demands
q = 3000		# [kg], cap of the truck
for i in lh[1:]:
	thisLHS += a*u[i]
model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=q, name='vehicle_cap_lh')

# Equation 8
thisLHS = LinExpr()
b = 500			# [kg], amount of product the bh customer offers
for i in bh[1:]:
	thisLHS += b*v[i] 
model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=q, name='vehicle_cap_bh')

# Equation 9
thisLHS = LinExpr()
for j in range(len(customers)):
	if j in lh[1:]:
		for i in range(len(customers)):
			thisLHS += x[i,j] 
		model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=u[j], name='flow_arrival_lh_customer_' + str(j))
	if j in bh:
		for i in range(len(customers)):
			thisLHS += x[i,j]
		model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=v[j], name='flow_arrival_bh_customer_' + str(j))

# Equation 10
thisLHS = LinExpr()
for i in range(len(customers)):
	if i in lh:
		for j in range(len(customers)):
			thisLHS += x[i,j]
		model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=u[i], name='flow_departure_lh_customer_' + str(j))
	if i in bh[1:]:
		for j in range(len(customers)):
			thisLHS += x[i,j]
		model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=v[i], name='flow_departure_bh_customer_' + str(j))

# Equation 11
thisLHS = LinExpr()
for i in lh:
	for j in bh:
		thisLHS += x[i,j]
model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=1, name='lh_priority')

# Equation 14
thisLHS = LinExpr()
for i in range(len(customers)):
	for j in range(len(customers)):
		if i !=j:
			thisLHS += f[j, i] - f[i, j]
	if i in lh:
		model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=a)
	if j in bh:
		model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=b)


# Equation 15 & 16
for i in range(len(customers)):
	for j in range(len(customers)):
		if i < 70:
			thislhs = a*x[i, j]
			thisrhs = (q-a)*x[i, j]
		else:
			thislhs = b*x[i, j]
			thisrhs = (q-b)*x[i, j]
		model.addConstr(lhs=thislhs, sense=GRB.LESS_EQUAL, rhs=f[i, j])
		model.addConstr(lhs=f[i,j], sense=GRB.LESS_EQUAL, rhs=thisrhs)
