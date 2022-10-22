#-------------------------------------------------------------------------
# AUTHOR: Siva Charan
# FILENAME: naive_bayes.py
# SPECIFICATION: naive bayes prediction accuracy
# FOR: CS 5990- Assignment #3
# TIME SPENT: 1 hr
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

classes = [i for i in range(-22, 40, 6)]
def discretize(instance):
    try:
        prev=-100
        for c in classes:
            if instance["Temperature (C)"] > prev and instance["Temperature (C)"] <= c:
                instance["Temperature (C)"] = c
            prev = c
    except:
        print("instance failed to discretize", instance)
    return instance

#11 classes after discretization

#reading the training data
#--> add your Python code here
training_df = pd.read_csv('weather_training.csv', nrows=11255) #works till 11255
training_df.dropna(how="all")
discrete_training_df = training_df.apply(discretize, axis = 1)

# The columns that we will be making predictions with.
y_training = np.array(discrete_training_df["Temperature (C)"])
X_training = np.array(discrete_training_df.drop(["Temperature (C)","Formatted Date"], axis=1).values)
#reading the test data
#--> add your Python code here
test_df = pd.read_csv('weather_test.csv')
test_df.dropna(how="all")

#update the test class values according to the discretization (11 values only)
discrete_test_df = test_df.apply(discretize, axis = 1)
y_test = discrete_test_df["Temperature (C)"]
X_test = discrete_test_df.drop(["Temperature (C)","Formatted Date"], axis=1).values


#fitting the naive_bayes to the data
clf = GaussianNB()
clf = clf.fit(X_training, y_training)

#make the naive_bayes prediction for each test sample and start computing its accuracy
#the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
#to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
num_of_accurate = 0
for (x_testSample, y_testSample) in zip(X_test, y_test):
    prediction = clf.predict(np.array([x_testSample]))
    #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
    #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
    diff = 100*(abs(prediction[0] - y_testSample)/y_testSample)
    if diff >= -15 and diff <= 15:
        num_of_accurate+=1

 
#print the naive_bayes accuracyy
#--> add your Python code here
#print("naive_bayes accuracy: " + str(accuracy))
score = num_of_accurate/len(y_test)
print(f"naive_bayes accuracy: {score}")





