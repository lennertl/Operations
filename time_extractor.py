''' 
* 	We get the weights necessary to account for congestion on the roads in the day 
	and the evening.
*	A certain fraction of the total travel time multiplied by the weight is added 
	to the real travel time. 
*	The real travel time matrix is determined from the distances in between the states
	to be visited.
'''

import os
from termcolor import colored
import pandas as pd

# Getting a dataframe from the csv file
os.chdir("../data")
cwd = os.getcwd()
instance_name = "VERKEERSPROGNOSE.csv"
df = pd.read_csv(os.path.join(cwd,instance_name), sep=';')

# Filtering the interesting elems: daglv, avondlv en nachtlv for etmaal >= 10000
min_days = 10000
df_reduced = df[["daglv","avondlv","nachtlv"]][df["etmaal"]>=min_days]

# function to calculate scaled number of cars from Lden [dBA], SEL_car ~= 82 dBA
def nr_cars(Lden):
	return 10**(Lden/1000)

# Make a new data frame with a scaled number of cars (multiple mic's across 1 street)
df_cars = pd.concat([df["naam"][df["etmaal"]>=min_days],nr_cars(df_reduced)], axis=1).dropna().sort_values("naam").reset_index()

# Taking the average over the data points per street
def reduce_datapts(df):
	counter = 0
	index = 1
	dfs = list()
	while index <= df_cars.shape[0]-1:
		if df_cars["naam"].iloc[index-1] == df_cars["naam"].iloc[index]:
			counter += 1
		else:
			dfs.append(df_cars.iloc[index-1-counter:index, -3:].mean())
			counter = 0
		index += 1
	# last elem is not appended since it appends only when there is a shift							
	dfs.append(df_cars.iloc[index-1-counter:index, -3:].mean())
	return index, counter, dfs

index, counter, dfs = reduce_datapts(df_cars)
mean_matrix = pd.concat(dfs, axis=1).T
matrix = pd.concat([df_cars["naam"].drop_duplicates().reset_index(),mean_matrix], axis = 1)

# determine normalization w.r.t. night (no congestion) 
def norm_col(df,col):
	return (df[col] - df["nachtlv"].min())/(df["nachtlv"].max()-df["nachtlv"].min())		# all centered and scaled according to night cond (no congestion)

# The mean() and std() of these can be used to model normal distributions to determine the weights for the time of day
day_norm = norm_col(mean_matrix,"daglv")
evening_norm = norm_col(mean_matrix,"avondlv")
night_norm = norm_col(mean_matrix,"nachtlv")

print(f"Day time matrix weight distribution: \t \t mu = {colored(round(day_norm.mean(),3),'green')}, sigma = {colored(round(day_norm.std(),3),'green')}")
print(f"Evening time matrix weight distribution: \t mu = {colored(round(evening_norm.mean(),3),'green')}, sigma = {colored(round(evening_norm.std(),3),'green')}")
print(f"Night time matrix weight distribution: \t \t mu = {colored(round(night_norm.mean(),3),'green')}, sigma = {colored(round(night_norm.std(),3),'green')}")