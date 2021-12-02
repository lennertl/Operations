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

# function to calculate numbber of cars from Lden [dBA], SEL_car ~= 82 dBA
def nr_cars(Lden):
	return 10**(Lden/1000)

# Make a new data frame with day/evening and day/night car densities, scaled
df_cars = nr_cars(df_reduced)

mean_day = df_cars["daglv"].mean()
mean_evening = df_cars["avondlv"].mean()
mean_night = df_cars["nachtlv"].mean()

print("Assuming that in the night (2200-0700), a weight factor of 1 should be included:")
print(f"In the day (0700-1900), on average, there are {colored(round(mean_day/mean_night,1),'green')} more cars than in the night.")
print(f"In the evening (1900-2200), on average, there are {colored(round(mean_evening/mean_night,1),'green')} more cars than in the night.")

