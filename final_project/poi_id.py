#!/usr/bin/python
""" A script that loads data prepared by Katie Malone (Udacity)
    relating to the Enron scandal, including financial and email
    records for 145 individuals, and creates a decision tree classifier
    that attempts to classify individuals based on whether they
    were considered a 'person-of-interest' ("POI") or not.
    Author: Mark Bannister (mspbannister@gmail.com)
"""

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# Initial features list
# features_list = ['poi','total_payments', 'total_stock_value',
#                  'from_poi_to_this_person', 'from_this_person_to_poi',
#                  'to_messages', 'from_messages', 'from_poi_rate', 'to_poi_rate']

# Final features list
features_list = ['poi','total_payments', 'total_stock_value',
                 'from_poi_to_this_person', 'to_poi_rate']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0) # Contains column total data
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0) # not an individual
data_dict.pop('LOCKHART EUGENE E', 0) # record contains no information
data_dict.pop('HUMPHREY GENE E', 0) # 'to_poi_rate' outlier
data_dict.pop('LAVORATO JOHN J', 0) # 'from_poi_to_this_person' / 'total_payments' outlier
data_dict.pop('FREVERT MARK A', 0) # 'total_payments' outlier

### Task 3: Create new feature(s)
my_dataset = {}
for key in data_dict:
    my_dataset[key] = data_dict[key]
    try:
        from_poi_rate = 1. * data_dict[key]['from_poi_to_this_person'] / \
        data_dict[key]['to_messages']
    except:
        from_poi_rate = "NaN"
    try:
        to_poi_rate = 1. * data_dict[key]['from_this_person_to_poi'] / \
        data_dict[key]['from_messages']
    except:
        to_poi_rate = "NaN"
    my_dataset[key]['from_poi_rate'] = from_poi_rate
    my_dataset[key]['to_poi_rate'] = to_poi_rate

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, remove_NaN=True, remove_all_zeroes=True, 
                     remove_any_zeroes=True, sort_keys=True)

labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier()

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier()
    
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# from sklearn.grid_search import GridSearchCV
# param_grid = {'criterion': ['gini', 'entropy'],
#               'min_samples_split': [2, 4, 6, 8],
#               'max_features': [2, 3, 4],
#               'max_depth': [3, 4, 5, None],
#               'n_estimators': [5, 10, 15, 20]}

# from sklearn.tree import DecisionTreeClassifier
# algo = DecisionTreeClassifier()

# from sklearn.ensemble import RandomForestClassifier
# algo = RandomForestClassifier()
#
# clf = GridSearchCV(algo, param_grid)

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(max_features=3, min_samples_split=4,
#                               criterion='entropy', max_depth=3,
#                               n_estimators=5)

# Create final classifer
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_features=2, min_samples_split=2,
                             criterion='entropy', max_depth=None)

# Test using Udacity testing function
# print test_classifier(clf, my_dataset, features_list, 1000)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)