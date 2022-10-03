# -------------------------------------------------------------------------
# AUTHOR: Siva Charan Mallena
# FILENAME: Decision tree classifier
# SPECIFICATION: Decision tree and average accuracy calculations
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: ~2 hr
# -----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv']
marital_status = {"Single":[1,0,0], "Divorced":[0,1,0], "Married":[0,0,1] }
refund = {"Yes":1, "No":0}
def process_instance_row(instance):
    new_instance = []
    new_instance.append(refund[instance[0]])
    hot_encode = marital_status[instance[1]]
    new_instance.append(hot_encode[0])
    new_instance.append(hot_encode[1])
    new_instance.append(hot_encode[2])
    taxable = instance[2].replace('k','')
    taxable = float(taxable)
    new_instance.append(taxable)
    return new_instance

for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:,1:] #creating a training matrix without the id (NumPy library)

    #transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    #Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    #be converted to a float.
    
    for instance in data_training:
        new_instance = process_instance_row(instance)
        X.append(new_instance)
        Y.append(refund[instance[3]])
    #print("X",X)
    # X =

    #transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    #print("Y",Y)

    #loop your training and test tasks 10 times here
    dtest = pd.read_csv('cheat_test.csv', sep=',', header=0)
    data_test = np.array(dtest.values)[:,1:]
    accuracies = []
    for i in range (10):

        #fitting the decision tree to the data by using Gini index and no max_depth
        clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
        clf = clf.fit(X, Y)

        #plotting the decision tree
        tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
        plt.show()

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for data in data_test:
           #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
           #class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           transformed_test_instance = process_instance_row(data)
           
           class_predicted = clf.predict([transformed_test_instance])[0]

           #compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
           test_class_value = refund[data[3]]
           if class_predicted == 1 and test_class_value == 1:
               tp+=1
           elif class_predicted == 1 and test_class_value == 0:
               fp+=1
           elif class_predicted == 0 and test_class_value == 1:
               fn+=1
           else:
               tn+=1
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        #print("ACCURACY",accuracy)
        accuracies.append(accuracy)
    print("accuracies", accuracies)
       #find the average accuracy of this model during the 10 runs (training and test set)
    avg_accuracy = np.average(accuracies)

    #print the accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    print("average accuracy when training on "+str(ds), avg_accuracy)




