#!/usr/bin/python

import pickle
import numpy
from time import time
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )


### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()
# print vectorizer.get_feature_names()[21323]

### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here
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
for i, value in enumerate(clf.feature_importances_):
    if value > 0.2:
        print i, value
        



