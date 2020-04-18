####################################################
# IMPORTING THE LIBRARIES                          #
####################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

####################################################
# DEFINE HELPER FUNCTIONS                          #
####################################################
def print_accuracy_report(strategy, result, prediction):
    no_of_tests = prediction.shape[0]
    no_of_correct = 0
    for i in range(0, no_of_tests):
        if result[i] == prediction[i]:
            no_of_correct += 1
    print(f"The accuracy of {strategy} is {no_of_correct / no_of_tests * 100}%")

####################################################
# IMPORTING THE TRAINING SET                       #
####################################################
dataset = pd.read_csv('training_set.csv')
# signature is a set of measurements (Temperature, Humidity and CO2 used to determine the occupancy
signature = dataset.iloc[:, :-1].values
# occupancy is an indicator to determine if the room is occupied (0 = NO and 1 = YES)
occupancy = dataset.iloc[:, -1].values

####################################################
# TAKING CARE OF MISSING DATA                      #
####################################################
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(signature[:, 0:3])
signature[:, 0:3] = imputer.transform(signature[:, 0:3])

####################################################
# FEATURE SCALING                                  #
####################################################
sc = StandardScaler()
signature = sc.fit_transform(signature)