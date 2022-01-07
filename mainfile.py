'''
Simplifications: only 1 veh, only daytime is considered atm, same time windows, same customer demands
'''

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
import matplotlib.pyplot as plt

# get distances df
os.chdir("../data")
cwd = os.getcwd()

#########################################################################################
################################### MODEL SETUP #########################################
#########################################################################################

# get the instance name that is corresponding to the data
instance_name = 'grid_compact.csv'

distance_name = 'distances.csv'
daytime_name = 'day_timematrix.csv'
eveningtime_name = 'evening_timematrix.csv'
nighttime_name = 'night_timematrix.csv'

distance_frame = pd.read_csv(os.path.join(cwd,distance_name))
daytime_frame = pd.read_csv(os.path.join(cwd,daytime_name))
eveningtime_frame = pd.read_csv(os.path.join(cwd,eveningtime_name))
nighttime_frame = pd.read_csv(os.path.join(cwd,nighttime_name))

distance = distance_frame.iloc[:,1:].to_numpy()
daytime = daytime_frame.iloc[:,1:].to_numpy()

# load the data for this instance
customers = pd.read_csv(instance_name)

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
lh = np.append([0],np.where(customers.iloc[:,3].to_numpy() == False)[0])	# depot needs to be available for lh and bh
# List of backhaul customers
bh = np.append([0],np.where(customers.iloc[:,3].to_numpy() == True)[0])		# depot needs to be available for lh and bh

# Binary decision variable =1 if customer is served
# Linhaul customers
u = {}	
for i in lh:
	u[i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name="u[%s]"%(i))

# Backhaul customers
v = {}
for i in bh:
	v[i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name="v[%s]"%(i))	
    
# time represented as integer variables (seconds), ub = 1 day
# depot needs to have a time as well = 0
# time at which service starts
t = {}
for i in range(len(customers)):
	t[i] = model.addVar(lb=0, ub=86400, vtype=GRB.INTEGER, name="t[%s]"%(i))

# load carried represented as integer variables (kg), ub = 3000
f = {}
for i in range(len(customers)):
	for j in range(len(customers)):
		f[i,j] = model.addVar(lb=0, ub=3000, vtype=GRB.INTEGER, name="f[%s,%s]"%(i,j))

model.update()

#########################################################################################
################################### CONSTRAINTS #########################################
#########################################################################################

depot = 43		# after doing np.sqrt((x-x0)**2+(y-y0)**2) for all nodes

# equation 5
for i in lh[1:]:
	model.addConstr(lhs=u[i], sense=GRB.EQUAL, rhs=1, name='customer_served_lh_' + str(i))

# equation 6
for i in bh[1:]:
	model.addConstr(lhs=v[i], sense=GRB.EQUAL, rhs=1, name='customer_served_bh_' + str(i))

# Equation 7
thisLHS = LinExpr()
a = 1000		# [kg], amount of product the lh customer demands
q = 3000		# [kg], cap of the truck
for i in lh[1:]:
	thisLHS += a*u[i]
model.addConstr(lhs=thisLHS, sense=GRB.LESS_EQUAL, rhs=q, name='vehicle_cap_lh')

# Equation 8
thisLHS = LinExpr()
b = 500			# [kg], amount of product the bh customer offers
for i in bh[1:]:
	thisLHS += b*v[i] 
model.addConstr(lhs=thisLHS, sense=GRB.LESS_EQUAL, rhs=q, name='vehicle_cap_bh')

# Equation 9
for j in range(len(customers)):
	if j in lh[1:]:
		thisLHS = LinExpr()
		for i in range(len(customers)):
			thisLHS += x[i,j] 
		model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=u[j], name='flow_arrival_lh_customer_' + str(j))
	if j in bh:
		thisLHS = LinExpr()
		for i in range(len(customers)):
			thisLHS += x[i,j]
		model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=v[j], name='flow_arrival_bh_customer_' + str(j))

# Equation 10
for i in range(len(customers)):
	if i in lh:
		thisLHS = LinExpr()
		for j in range(len(customers)):
			thisLHS += x[i,j]
		model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=u[i], name='flow_departure_lh_customer_' + str(i))
	if i in bh[1:]:
		thisLHS = LinExpr()
		for j in range(len(customers)):
			thisLHS += x[i,j]
		model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=v[i], name='flow_departure_bh_customer_' + str(i))

# Equation 11
thisLHS = LinExpr()
for i in lh:
	for j in bh:
		thisLHS += x[i,j]
model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=1, name='lh_priority')

# Equation 12 
s = 300			# [s], service time = 5min
R = 2*86400		# [-], big-M variable, forces time to be 2*ub
for j in range(1,len(customers)):		# start time for arrival node
	for i in range(1,len(customers)):	# start time for departure node
		thisLHS = LinExpr()
		thisLHS += t[j] - t[i] - s - daytime[i][j] + (1-x[i,j])*R
		model.addConstr(lhs=thisLHS, sense=GRB.GREATER_EQUAL, rhs=0, name='start_time_dep_' + str(i) + '_arr_' + str(j))

# Equation 13
e = 3000		# [s], time window begin
l = 86400 - e 	# [s], time window end
for i in range(1,len(customers)):
	model.addConstr(lhs=e, sense=GRB.LESS_EQUAL, rhs=t[i], name='time_window_lhs_' + str(i))
	model.addConstr(lhs=t[i], sense=GRB.LESS_EQUAL, rhs=l, name='time_window_rhs_' + str(i))

# Equation 14
for i in range(len(customers)):
	if i in lh[1:]:
		thisLHS = LinExpr()
		for j in range(len(customers)):
			if i != j:
				thisLHS += f[j,i]*x[j,i] - f[i,j]*x[i,j]		# sum of sum elements = sum elements of sum (linear)
		model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=a, name='load_flow_balance_lh_' + str(i))
	if i in bh[1:]:
		thisLHS = LinExpr()
		for j in range(len(customers)):
			if i != j:
				thisLHS += f[j,i]*x[j,i] - f[i,j]*x[i,j]
		model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=b, name='load_flow_balance_bh_' + str(i))


thisLHS = LinExpr()
for j in range(1,len(customers)):
	thisLHS += f[0,j]*x[0,j]
model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=q, name='forced_depot_output')

thisLHS = LinExpr()
for i in range(1,len(customers)):
	thisLHS += f[i,0]*x[i,0]
model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=q-a-b, name='forced_depot_input')

# Equation 15
for j in range(1,len(customers)):
	for i in range(1,len(customers)):
		model.addConstr(lhs=a*x[i,j], sense=GRB.LESS_EQUAL, rhs=f[i,j], name='load_cap_lh_lhs_rte_' + str(i) + '_' + str(j))
		model.addConstr(lhs=f[i,j], sense=GRB.LESS_EQUAL, rhs=(q-a)*x[i,j], name='load_cap_lh_rhs_rte_' + str(i) + '_' + str(j))

# Equation 16
for j in range(1,len(customers)):
	for i in range(1,len(customers)):
		model.addConstr(lhs=b*x[i,j], sense=GRB.LESS_EQUAL, rhs=f[i,j], name='load_cap_bh_lhs_rte_' + str(i) + '_' + str(j))
		model.addConstr(lhs=f[i,j], sense=GRB.LESS_EQUAL, rhs=(q-b)*x[i,j], name='load_cap_bh_rhs_rte_' + str(i) + '_' + str(j))

# Equation 17-20 are integrality and binary constraints that are already incl in var declaration
# To enforce the depot to be a lh as well as a bh customer, constraints need to be implemented manually
model.addConstr(lhs=u[0], sense=GRB.EQUAL, rhs=1, name='depot_forced_lh_node')
model.addConstr(lhs=v[0], sense=GRB.EQUAL, rhs=1, name='depot_forced_bh_node')

model.update()

#########################################################################################
################################### FINALIZATION ########################################
#########################################################################################

# Init the obj function
obj = LinExpr()

# Defining the objective function
alpha = 9.80665*0.01 			# specific constant of arc, road friction drag
w = 7000.						# [kg], weight of the vehicle
beta = 0.5*0.7*6*1.2041			# specific vehicle constant, aerodynamic drag (assuming 2D)
V = 90./3.6 					# [m/s], assumed constant velocity
for i in range(len(customers)):
	for j in range(len(customers)):
		#obj += (alpha*distance[i][j]*(w*x[i,j] + f[i,j]) + beta*V**2*distance[i][j])
		obj += distance[i][j]*x[i,j]

# Telling the solver what should be done with the objective's value
model.setObjective(obj,GRB.MINIMIZE)
model.update()

# Writing the standardized .lp file from the generated model
model.write('mainfile_alt_model_formulation.lp')

# optimize the model 
model.optimize()

'''
# adding dummy variables to make it feasible
copy1 = model.copy()

if model.status == GRB.INFEASIBLE:
 copy1.feasRelaxS(1, True, False, True)
 copy1.optimize()
'''

# Overall performance eval
endTime = time.time()
print(f'The execution time is: {endTime-startTime}')

# Saving the solution (name of var, val of var)
solution = []		# init sol list
for v in model.getVars():		# iterate over all vars
	solution.append([v.varName,v.x])

print(solution)

## plotting
x_coord = customers['x']
y_coord = customers['y']
backhaul = customers['Backhaul']
plt.scatter(x_coord[0], y_coord[0], c="orange", label='depot')
backhaul_label = 'backhaul'
linehaul_label = 'linehaul'
for i in range(1, len(x_coord)):
	if backhaul[i]:
		plt.scatter(x_coord[i], y_coord[i], c="red", label=backhaul_label)
		backhaul_label = '__nolegend__'
	else:
		plt.scatter(x_coord[i], y_coord[i], c="green", label='linehaul')
		linehaul_label = '__nolegend__'
for i in range(len(customers)):
	for j in range(len(customers)):
		if x[i, j].x:
			plt.plot([x_coord[i], x_coord[j]], [y_coord[i], y_coord[j]], c='blue')
plt.legend()
plt.show()
