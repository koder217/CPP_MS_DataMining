#-------------------------------------------------------------------------
# AUTHOR: Siva Charan
# FILENAME: bagging_random_forest.py
# SPECIFICATION: Comparing ensemble methods
# FOR: CS 5990- Assignment #4
# TIME SPENT: 4 hrs
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

dbTraining = []
dbTest = []
X_training = []
y_training = []
classVotes = [0,0,0,0,0,0,0,0,0,0] #this array will be used to count the votes of each classifier

#reading the training data from a csv file and populate dbTraining
#--> add your Python code here
dbTraining = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

#reading the test data from a csv file and populate dbTest
#--> add your Python code here
dbTest = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library
total_test_samples = len(dbTest.index)
X_test = np.array(dbTest.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(dbTest.values)[:,-1]
#inititalizing the class votes for each test sample. Example: classVotes.append([0,0,0,0,0,0,0,0,0,0])
#--> add your Python code here

print("Started my base and ensemble classifier ...")

for k in range(20): #we will create 20 bootstrap samples here (k = 20). One classifier will be created for each bootstrap sample

  bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)
  
  #populate the values of X_training and y_training by using the bootstrapSample
  #--> add your Python code here
  X_training = np.array(bootstrapSample.values)[:,:64]
  y_training = np.array(bootstrapSample.values)[:,-1]
  
  #fitting the decision tree to the data
  clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=None) #we will use a single decision tree without pruning it
  clf = clf.fit(X_training, y_training)
  #print("dbTest", dbTest.shape)
  num_of_accurate = 0

  for i, testSample in enumerate(dbTest):
      #print(dbTest.iloc[i])
      #make the classifier prediction for each test sample and update the corresponding index value in classVotes. For instance,
      # if your first base classifier predicted 2 for the first test sample, 
      #     then classVotes[0,0,0,0,0,0,0,0,0,0] will change to classVotes[0,0,1,0,0,0,0,0,0,0].
      # Later, if your second base classifier predicted 3 for the first test sample, 
      #     then classVotes[0,0,1,0,0,0,0,0,0,0] will change to classVotes[0,0,1,1,0,0,0,0,0,0]
      # Later, if your third base classifier predicted 3 for the first test sample, 
      #     then classVotes[0,0,1,1,0,0,0,0,0,0] will change to classVotes[0,0,1,2,0,0,0,0,0,0]
      # this array will consolidate the votes of all classifier for all test samples
      #--> add your Python code here
      prediction = int(clf.predict([dbTest.iloc[i,:64]])[0])
      true_label = dbTest.iloc[i,-1]
      classVotes[prediction] += 1
      # print("prediction", prediction)
      # print("y_test", y_test)
      # print(np.mean(prediction == y_test))
      if k == 0: #for only the first base classifier, compare the prediction with the true label of the test sample here to start calculating its accuracy
          #--> add your Python code here
          if prediction == true_label:
              num_of_accurate += 1
          

  if k == 0: #for only the first base classifier, print its accuracy here
      #--> add your Python code here
      accuracy = num_of_accurate / total_test_samples
      print("Finished my base classifier (fast but relatively low accuracy) ...")
      print("My base classifier accuracy: " + str(accuracy))
      print("")

#now, compare the final ensemble prediction (majority vote in classVotes) 
#for each test sample with the ground truth label to calculate the accuracy of 
#the ensemble classifier (all base classifiers together)
#--> add your Python code here
ground_truth_label_counts = [0,0,0,0,0,0,0,0,0,0]
ground_truth_labels = dbTest.iloc[:,-1:]
for index in range(len(ground_truth_labels)):
    truth = int(ground_truth_labels.iloc[index])
    ground_truth_label_counts[truth] += 1

errors = 0
for i in range(10):
    errors += abs(classVotes[i] - ground_truth_label_counts[i])

accuracy = 1 - (errors / total_test_samples)
#printing the ensemble accuracy here
print("Finished my ensemble classifier (slow but higher accuracy) ...")
print("My ensemble accuracy: " + str(accuracy))
print("\nStarted Random Forest algorithm ...")

#Create a Random Forest Classifier
clf=RandomForestClassifier(n_estimators=20) #this is the number of decision trees that will be generated by Random Forest. The sample of the ensemble method used before

#Fit Random Forest to the training data
clf.fit(X_training,y_training)

#make the Random Forest prediction for each test sample. Example: class_predicted_rf = clf.predict([[3, 1, 2, 1, ...]]
#--> add your Python code here
accurate = 0
for x, y in zip(X_test, y_test):
    predicted = clf.predict([x])

#compare the Random Forest prediction for each test sample with the ground truth label to calculate its accuracy
#--> add your Python code here
    if predicted[0] == y:
        accurate += 1

accuracy = accurate / total_test_samples


#printing Random Forest accuracy here
print("Random Forest accuracy: " + str(accuracy))

print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")
