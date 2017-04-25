#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
print "No. features: "+str(len(features_train[0]))
from sklearn.tree import DecisionTreeClassifier
min_samples_split=40
clf = DecisionTreeClassifier(min_samples_split=min_samples_split)
print "Fitting model..."
t0 = time()
clf = clf.fit(features_train, labels_train)
print "Creating predictions..."
t1 = time()
pred = clf.predict(features_test)
t2 = time()
from sklearn.metrics import accuracy_score
print "Computing accuracy..."
acc = accuracy_score(pred, labels_test)
timeDiff = round(t1-t0, 3)
print "Training time: "+str(timeDiff)+"s"
timeDiff = round(t2-t1, 3)
print "Prediction time: "+str(timeDiff)+"s"
print "Accuracy: "+str(acc)
print sum(pred)
#########################################################


