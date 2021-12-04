import os
import pandas as pd
import numpy as np

# get distances df
os.chdir("../data")
cwd = os.getcwd()
instance_name = "distances.csv"
distance_frame = pd.read_csv(os.path.join(cwd,instance_name))

# get the distance matrix
distances = distance_frame.iloc[:,1:].to_numpy()

# convert to nominal time matrix
velocity = 90.		# km/h
nominal_time = distances/velocity		# h

# Random normal generator to model the addition in time due to congestion, scaled for realism
mu = [3.213,0.894,0.134]		# d,e,n
sigma = [9.377,1.206,0.16] 		# d,e,n

np.random.seed(1)
day_gen = abs(np.random.normal(mu[0],sigma[0],(100,100)))/100		# 07:00 - 19:00
evening_gen = abs(np.random.normal(mu[1],sigma[1],(100,100)))/100	# 19:00 - 22:00
night_gen = abs(np.random.normal(mu[2],sigma[2],(100,100)))/100		# 22:00 - 07:00

# get time ref matrices for d,e,n
day_time = np.add(nominal_time,np.multiply(day_gen,nominal_time))
evening_time = np.add(nominal_time,np.multiply(evening_gen,nominal_time))
night_time = np.add(nominal_time,np.multiply(night_gen,nominal_time))

# write out the dataframes in csv format
day_df = pd.DataFrame(day_time)
evening_df = pd.DataFrame(evening_time)
night_df = pd.DataFrame(night_time)

day_df.to_csv(os.path.join(cwd,"day_timematrix.csv"))
evening_df.to_csv(os.path.join(cwd,"evening_timematrix.csv"))
night_df.to_csv(os.path.join(cwd,"night_timematrix.csv"))