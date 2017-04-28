# DAND P5: ‘Identify Fraud from Enron Email’
**Mark Bannister**  
*April 2017*  

## Introduction

Enron was an American energy trading and utilities company that, at its peak in 2000, [was worth over $66bn](https://www.theguardian.com/business/2001/nov/29/corporatefraud.enron1). Little more than a year later the company filed for bankruptcy, [mired in scandal](https://en.wikipedia.org/wiki/Enron_scandal). As a result, ‘Enron’ today is a byword for corporate fraud. 

In the criminal investigation that ensued, a large volume of confidential information entered the public domain, including email and financial records for several Enron executives. Using the scikit-learn machine learning package, I hope to create a program that can predict which of these executives were “persons-of-interest” – that is, individuals who were indicted, reached a settlement with the authorities, or testified in exchange for immunity from prosecution.

## Data exploration

>  *Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?*

The goal of this project is to investigate whether we can use machine learning to create a program (“classifier”) that can predict whether an individual at Enron was involved in the Enron scandal (i.e. whether they were “persons-of-interest” or “POIs”), based on financial and/or email data available in the public domain. The data set contains information on 146 people connected with Enron, including salary and bonus data, the value of Enron stock held by the individual, the number of emails sent to/from that individual, and whether the individual was known to be a POI.

Machine learning can help us spot patterns across multiple variables that enable more accurate classification than could be achieved by manual means. As far as this project is concerned, that might mean that, for example, salary may not accurately predict whether an individual was a POI, but when combined with stock holding and email information, a clearer trend may appear.

The data set contains a few potential outliers. Some individuals (e.g. Kenneth Lay) have salaries and/or stock positions vastly greater than the majority. However, considering their role in the scandal, I believe these to be valid data points. There are two records that do not relate to individuals at all: 

* ```'TOTAL'```, containing the sum of all the financial information in the data set; and
* ```'THE TRAVEL AGENCY IN THE PARK'```, which is clearly not an individual.

One of the other records, ```'LOCKHART EUGENE E'```, does not contain any information whatsoever. I removed all three of these entries using Python’s built-in ```'pop'``` function, reducing the data set to 143 records.

The main issue with the data is completeness; of 143 records, only 52 of them contain complete financial and email information for the features I have chosen to examine in my final analysis. This is broken down as follows (% indicates proportion of the two respective data sets):

```
Full data  
=========  
Number of data points: 143  
    POIs: 18 (12.59%)  
    Non-POIs: 125 (87.41%)  
'total_payments' missing: 20 (13.99%)  
'total_stock_value' missing: 18 (12.59%)  
'from_poi_to_this_person' missing: 57 (39.86%)  
'from_this_person_to_poi' missing: 57 (39.86%)  
'to_messages' missing: 57 (39.86%)  
'from_messages' missing: 57 (39.86%)  
  
Final analysis  
==============    
Number of data points: 52  
    POIs: 14 (26.92%)  
    Non-POIs: 38 (73.08%)    
```

By using only records containing complete information, we reduce the amount of data that we can use to train our classifier, which will inevitably impact its accuracy.

Not only is much of the data incomplete, but the significant disparity between the number of POIs and non-POIs may result in problems when training a classification algorithm to distinguish between the two (an effect known as [class imbalance](http://www.chioka.in/class-imbalance-problem/)). 

Ultimately, these two issues may limit the usefulness of our final classifier.

## Feature selection

> *What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.*

I have chosen to focus on the following features: 
* ```'total_payments'``` (i.e. total salary and bonus payments)
* ```'total_stock_value'```
* ```'from_poi_to_this_person'``` (no. of emails received from known POIs)
* ```'from_this_person_to_poi'``` (no. of emails sent to known POIs)
* ```'to_messages'``` (total emails received)
* ```'from_messages'``` (total emails sent)

I also created two new features:
* ```'from_poi_rate'``` (no. of emails received from known POIs as proportion of total emails received)
* ```'to_poi_rate'``` (no. of emails sent to known POIs as proportion of total emails sent)

I based these choices on the following intuition:
* Complex fraud requires collaboration and communication, which could be demonstrated through higher than normal rates of emails being sent to/from POIs from other POIs.
* Individuals are more likely to commit fraud if the (potential) financial rewards outweigh the risk of getting caught. Thus, I would expect to see POIs receiving relatively greater salary and/or bonus payments than non-POIs.
* Similarly, POIs may have been rewarded with increased stock options, which would result in them having greater total stock values. We might also consider that individuals with greater stock positions to begin with would be interested in maximising the value of the company, potentially by colluding to artificially inflate the company’s profits, as was the case in the Enron scandal.

I fit the original six features to a decision tree classifier in scikit-learn (using default parameter values) and tested their performance using the ```'tester.py'``` script. I then repeated the test with each of my new features, yielding the following results: 

```
Original features: Accuracy: 0.79664, Precision: 0.28836, Recall: 0.28850
w/'from_poi_rate': Accuracy: 0.79307, Precision: 0.26988, Recall: 0.26300
w/'to_poi_rate': Accuracy: 0.80736, Precision: 0.30454, Recall: 0.27150
w/both new features: Accuracy: 0.80386, Precision: 0.29595, Recall: 0.27050
```

We can see that adding ```'to_poi_rate'``` improved accuracy and precision, but reduced the classifier’s recall. Adding ```'from_poi_rate'``` appeared to reduce performance both when it was added on its own, and when it was added in combination with ```'to_poi_rate'```.

To select which of these features to use in my classifier, I fitted them all to a decision tree and used the ```.feature_importances_``` attribute to determine which features were most influential. This produced the following results (averaged over 100 iterations):

```
total_payments: 0.186683494368  
total_stock_value: 0.179933409987  
from_poi_to_this_person: 0.135166356563  
from_this_person_to_poi: 0.0379356126231  
to_messages: 0.0692685600811  
from_messages: 0.0889360557329  
from_poi_rate: 0.0889379547817  
to_poi_rate: 0.213138555863   
```

Given the relatively similar levels of importance attributed to ```'total_payments'```, ```'total_stock_value'```, ```‘from_poi_to_this_person'``` and ```'to_poi_rate'```, I decided to explore these features further. After visualising them, I chose to remove the following outliers:

* ```'HUMPHREY GENE E'```: a ```'to_poi_rate'``` outlier
* ```'LAVORATO JOHN J'```: a ```'from_poi_to_this_person' ```/ ```'total_payments'``` outlier
* ```'FREVERT MARK A'```: a ```'total_payments'``` outlier

I then repeated the ```.feature_importances_```  exercise with only these four features, and obtained the following results:

```
total_payments: 0.262814037552
total_stock_value: 0.400085932203
from_poi_to_this_person: 0.093454782616
to_poi_rate: 0.243645247629
```

Given this now seems to suggest that ```'from_poi_to_this_person'``` is significantly less important than the other three features, I decided to test two combinations of these features using a default decision tree classifier and the ```'tester.py'``` script: one including ```'from_poi_to_this_person'```, and one excluding it. The results were as follows:

```
w/'from_poi_to_this_person': Accuracy: 0.81564, Precision: 0.32677, Recall: 0.27400
w/out 'from_poi_to_this_person': Accuracy: 0.80707, Precision: 0.29960, Recall: 0.26200
```

While ```'from_poi_to_this_person'``` may not be as important as the other three features, it seems to improve the accuracy of the classifier, so I chose to retain all four features in my final analysis.

## Algorithm selection and tuning

> *What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?*

I tried three different classification algorithms: Naïve Bayes (“NB”), decision tree (“DT”) and random forest (“RF”). None of these algorithms require feature scaling, so I used the original data values. Using each classifier’s default parameter settings, I observed the following performance using the ```'tester.py'``` script:
```
NB: Accuracy: 0.85507, Precision: 0.48488, Recall: 0.23250
DT: Accuracy: 0.81564, Precision: 0.32677, Recall: 0.27400
RF: Accuracy: 0.84429, Precision: 0.40405, Recall: 0.18950
```
For out-of-the-box performance, NB appears to be the strongest overall, while DT was the most balanced. 

> *What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm?*

Tuning the parameters of an algorithm is the process of optimising those parameters to enable maximum performance (i.e. accuracy) on the problem being investigated. The parameters control the algorithm’s response to training data, and help ensure it reflects the shape of the data. 

For example, the ‘kernal’ parameter in a support vector machine classifier determines whether the algorithm tries to impose a linear or non-linear decision boundary. If a linear decision boundary is applied to non-linear data, the algorithm is unlikely to produce accurate classifications. Correctly tuned parameters can also guard against over-fitting, e.g. by making sure a decision tree classifier doesn’t create separate branches for every individual outcome.

While the NB algorithm performed strongly in my initial testing, it has less scope for parameter optimisation than the other two algorithms. I tuned the DT and RF algorithms using scikit-learn’s ```GridSearchCV``` function, which tested different combinations of ```'criterion'```, ```'min_samples_split'```, ```'max_features'```, ```'max_depth'``` and (for RF) ```'n_estimators'``` as follows:

```
'criterion': ['gini', 'entropy'],
'min_samples_split': [2, 4, 6, 8],
'max_features': [2, 3, 4],
'max_depth': [3, 4, 5, None]
'n_estimators': [5, 10, 15, 20]
```

The best results I was able to achieve were as follows:

```
DT: Accuracy: 0.82007, Precision: 0.35397, Recall: 0.31450
RF: Accuracy: 0.84271, Precision: 0.40578, Recall: 0.21750
```

Based on these results, I decided to use DT in my final analysis, as it provided the most balanced performance between evaluation metrics and achieved the minimum recall of 0.3 (as specified in the project brief). The final parameter settings were as follows:

```
max_features=2,  
min_samples_split=2,  
criterion='entropy',   
max_depth=None  
```

## Validation

> *What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?*

Validation is the process of retaining a sample of the data set and using it to test the classifier once it has been tuned and trained. This is important, because if the classifier is only tuned using training and test sets, it may become overfitted to the test set and thus underperform in real life applications. The validation set therefore acts as a final check to ensure overfitting has not occurred. 

With small data sets such as the Enron set, the data sampling process that creates the training, test and validation sets can have a significant impact on the classifier’s performance – for example, if the distribution of data in the training set does not reflect that of the wider set. To overcome this, I used a cross-validation function, which randomly splits the data into k samples and trains the classifier on each of the k-1 samples, before validating it on the remaining data. The classifier’s performance is thus averaged across each of the samples. 

The specific function I used (```StratifiedShuffleSplit```) has the additional benefit of stratifying each random sample, such that the distribution of classes (i.e. POI and non-POI) in each sample reflects that of the larger data set. This is important, particularly in such a small and unevenly distributed data set, because otherwise there is no guarantee that each sample being used to train the classifier actually contains POI data for it to learn from.

## Evaluation

> *Give at least 2 evaluation metrics and your average performance for each of them. Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.*

I evaluated my classifier primarily using the ‘precision’ and ‘recall’ metrics. [Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall) defines these as measures of “exactness” and “completeness”, respectively. In the context of this investigation, precision measures the proportion of correctly identified POIs out of the total number of individuals classified as POIs by our classifier. Recall measures the proportion of correctly identified POIs out of the total number of POIs in the data set.

Using the ```'tester.py'``` script, averaged over 100 iterations, my classifier achieved the following performance:

```
Precision: 0.351787348863, Recall: 0.3083
```
Therefore, approximately 35.2% of the individuals identified as POIs by the classifier were, in fact, POIs, which corresponded to 30.8% of the overall number of POIs in the data set. Clearly this is not particularly accurate, which would impact its usefulness on any similar investigations in future. 

As demonstrated earlier, one of the features I developed during this investigation (```'to_poi_rate'```) improved the performance of my classifier. It is possible that other potential features lie in the data set that would lead to more accurate classifications. For example, we might search all the emails for the presence of key words or phrases, e.g. “[special purpose entities]( https://en.wikipedia.org/wiki/Enron_scandal#Special_purpose_entities)” and determine the frequency that they were used by individuals. I would be interested in exploring this further in a future investigation.

## Declaration

I hereby confirm that this submission is my work. I have cited above the origins of any parts of the submission that were taken from Websites, books, forums, blog posts, github repositories, etc.

## List of resources used

* http://scikit-learn.org/stable/modules/feature_selection.html  
* http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
* http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
* http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html  
* http://scikit-learn.org/stable/modules/cross_validation.html  
* http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html