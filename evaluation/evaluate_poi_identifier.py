#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
from time import time
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42)
# min_samples_split=40
clf = DecisionTreeClassifier()#min_samples_split=min_samples_split)
print "Fitting model..."
t0 = time()
clf = clf.fit(features_train, labels_train)
print "Creating predictions..."
t1 = time()
pred = clf.predict(features_test)
t2 = time()
from sklearn.metrics import accuracy_score, precision_score, recall_score
print "Computing accuracy..."
acc = accuracy_score(labels_test, pred)
prec = precision_score(labels_test, pred)
rec = recall_score(labels_test, pred)
timeDiff = round(t1-t0, 3)
print "Training time: "+str(timeDiff)+"s"
timeDiff = round(t2-t1, 3)
print "Prediction time: "+str(timeDiff)+"s"
print "Accuracy: "+str(acc)
print "Precision: "+str(prec)
print "Recall: "+str(rec)
