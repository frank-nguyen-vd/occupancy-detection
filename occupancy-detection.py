####################################################
# IMPORTING THE LIBRARIES                          #
####################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

####################################################
# IMPORTING THE TRAINING SET                       #
####################################################
dataset = pd.read_csv('training_set.csv')
# signature is a set of measurements (Temperature, Humidity and CO2 used to determine the occupancy
signature = dataset.iloc[:, :-1].values
# occupancy is an indicator to determine if the room is occupied (0 = NO and 1 = YES)
occupancy = dataset.iloc[:, -1].values

