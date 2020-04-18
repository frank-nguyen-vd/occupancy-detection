####################################################
# IMPORTING THE LIBRARIES                          #
####################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

####################################################
# DEFINE SYSTEM PARAMETERS                         #
####################################################
TRAINING_DATA = "training_set.csv"
TESTING_DATA  = "test_set.csv"

####################################################
# DEFINE HELPER FUNCTIONS                          #
####################################################
def print_accuracy_report(strategy, result, prediction):
    no_of_tests = prediction.shape[0]
    no_of_correct = 0
    for i in range(0, no_of_tests):
        if result[i] == prediction[i]:
            no_of_correct += 1
    print(f"The accuracy of {strategy} is {round(no_of_correct / no_of_tests * 100, 2)}%")

def import_dataset(path):
    dataset = pd.read_csv(path)
    # condition is a set of measurements (Temperature, Humidity and CO2 used to determine the occupancy
    condition = dataset.iloc[:, :-1].values
    # result is an indicator to determine if the room is occupied (0 = NO and 1 = YES)
    result = dataset.iloc[:, -1].values

    # taking care of missing data
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(condition[:, 0:3])
    condition[:, 0:3] = imputer.transform(condition[:, 0:3])

    # feature scaling
    sc = StandardScaler()
    condition = sc.fit_transform(condition)    

    return condition, result

####################################################
# IMPORTING THE DATA SET                           #
####################################################
train_condition, train_result = import_dataset(TRAINING_DATA)
test_condition, test_result   = import_dataset(TESTING_DATA)

####################################################
# DECISION TREE REGRESSION MODEL                   #
####################################################

# Training the model
model = DecisionTreeRegressor(random_state=0)
model.fit(train_condition, train_result)

# Testing the model
test_prediction = model.predict(test_condition)

# Printing the report
print_accuracy_report("DECISION TREE REGRESSION MODEL", test_result, test_prediction)

####################################################
# RANDOM FOREST REGRESSION MODEL                   #
####################################################
# Step 1: Pick at random K data points from the Training set
# Step 2: Build the Decision Tree associated to these K data points
# Step 3: Using Step 1 & 2 to build N decision trees
# Step 4: For a new data point, make each one of N trees predict
#         a value for the new data point. The final prediction is
#         the average of N predicted values

#  Training the model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(train_condition, train_result)

# Testing the model
test_prediction = model.predict(test_condition)

# Printing the report
print_accuracy_report("RANDOM FOREST REGRESSION MODEL", test_result, test_prediction)

####################################################
# LOGISTIC REGRESSION MODEL                        #
####################################################

#  Training the model
model = LogisticRegression(random_state=0)
model.fit(train_condition, train_result)

# Testing the model
test_prediction = model.predict(test_condition)

# Printing the report
print_accuracy_report("LOGISTIC REGRESSION MODEL", test_result, test_prediction)

####################################################
# K-NEAREST-NEIGHBOURS                            #
####################################################

# Training the model
model = KNeighborsClassifier(n_neighbors=100, weights='distance', metric='minkowski', p=2)
model.fit(train_condition, train_result)

# Testing the model
test_prediction = model.predict(test_condition)

# Printing the report
print_accuracy_report("K-NEAREST-NEIGHBOURS", test_result, test_prediction)

####################################################
# SUPPORT VECTOR MACHINE                           #
####################################################

# Training the model
model = SVC(kernel = 'sigmoid', random_state = 0)
model.fit(train_condition, train_result)

# Testing the model
test_prediction = model.predict(test_condition)

# Printing the report
print_accuracy_report("SUPPORT VECTOR MACHINE", test_result, test_prediction)
