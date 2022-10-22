#-------------------------------------------------------------------------
# AUTHOR: Siva Charan Mallena
# FILENAME: knn.py
# SPECIFICATION: K neighbour regressor
# FOR: CS 5990- Assignment #3
# TIME SPENT: 3 hrs
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np

#defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['distance', 'uniform']

#reading the training data
training_df = pd.read_csv('weather_training.csv')
training_df.dropna(how="any")
# The columns that we will be making predictions with.
y_training = np.array(training_df["Temperature (C)"])
X_training = np.array(training_df.drop(["Temperature (C)","Formatted Date"], axis=1).values)

#reading the test data
test_df = pd.read_csv('weather_test.csv')
test_df.dropna(how="any")
#The column that we want to predict.
y_test = test_df["Temperature (C)"]
X_test = test_df.drop(["Temperature (C)","Formatted Date"], axis=1).values
#hint: to convert values to float while reading them -> np.array(df.values)[:,-1].astype('f')

#loop over the hyperparameter values (k, p, and w) ok KNN
#--> add your Python code here
highest_accuracy = 0
for k in k_values:
    for p in p_values:
        for w in w_values:
            num_of_accurate = 0
            #fitting the knn to the data
            clf = KNeighborsRegressor(n_neighbors=k, p=p, weights=w)
            clf = clf.fit(X_training, y_training)

            #make the KNN prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously, use zip()
            for (x_testSample, y_testSample) in zip(X_test, y_test):
               prediction = clf.predict(np.array([x_testSample]))
               #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
               #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
               diff = 100*(abs(prediction[0] - y_testSample)/y_testSample)
               if diff >= -15 and diff <= 15:
                   num_of_accurate+=1

            
               #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
               #with the KNN hyperparameters. Example: "Highest KNN accuracy so far: 0.92, Parameters: k=1, p=2, w= 'uniform'"
            score = num_of_accurate/len(y_test)
            if score > highest_accuracy:
                print(f"Highest KNN accuracy so far: {score}, Parameters: k={k}, p={p}, w= '{w}'")
                highest_accuracy = score





