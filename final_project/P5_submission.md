# DAND P5: ‘Identify Fraud from Enron Email’
## Mark Bannister
### April 2017

## Introduction

*Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?*

The goal of this project is to use machine learning to create a program (“classifier”) that can predict whether an individual at Enron was involved in the [Enron scandal](https://en.wikipedia.org/wiki/Enron_scandal) (“persons of interest” or “POIs”), based on financial and/or email data available in the public domain. The data set contains information on 145 people connected with Enron, including salary and bonus data, the value of Enron stock held by the individual, the number of emails sent to/from that individual, and whether the individual was known to be a POI.

Machine learning can help us spot patterns across multiple variables that enable more accurate classification than could be achieved by manual means. As far as this project is concerned, that might mean that, for example, salary may not accurately predict whether an individual was a POI, but when combined with stock holding and email information, a clearer trend may appear.

The data set contains a few potential outliers. Some individuals (e.g. Kenneth Lay) have salaries and/or stock positions vastly greater than the majority. However, considering their role in the scandal, I believe these to be valid data points. The data does however include a ‘total’ row, containing the sum of all the financial information in the data set, which I have removed (using Python’s built-in ‘pop’ function). 

The main issue with the data is completeness; of 145 records, only 57 of them contain complete financial and email information for the features I have chose to examine in my final analysis. This is broken down as follows (% indicates proportion of the two respective data sets):

```
Full data  
=========  
Number of data points: 145       
    POIs: 18 (12.41%)  
    Non-POIs: 127 (87.59%)  
'total_payments' missing: 21 (14.48%)  
'total_stock_value' missing: 20 (13.79%)  
'from_poi_to_this_person' missing: 59 (40.69%)  
'from_this_person_to_poi' missing: 59 (40.69%)  
'to_messages' missing: 59 (40.69%)  
'from_messages' missing: 59 (40.69%)    

Complete records only  
=====================  
Number of data points: 57  
   POIs: 14 (24.56%)  
   Non-POIs: 43 (75.44%)  
```

Having less data available to train our classifier will inevitably impact its accuracy. However, with the number of POIs now representing a greater proportion of the data, clearer patterns may emerge that will distinguish between the two classes.

## Feature selection

*What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.*

I have chosen to focus on the following features: 
* 'total_payments' (i.e. total salary and bonus payments)
* 'total_stock_value'
* 'from_poi_to_this_person' (no. of emails received from known POIs)
* 'from_this_person_to_poi' (no. of emails sent to known POIs)
* 'to_messages' (total emails received)
* 'from_messages' (total emails sent)

I also created two new features:
* 'from_poi_rate' (no. of emails received from known POIs as proportion of total emails received)
* 'to_poi_rate' (no. of emails sent to known POIs as proportion of total emails sent)

I based these choices on the following intuition:
* Complex fraud requires collaboration and communication, which could be demonstrated through higher than normal rates of emails being sent to/from POIs from other POIs.
* Individuals are more likely to commit fraud if the (potential) financial reward outweighs the risk of getting caught. Thus, I would expect to see POIs receiving relatively greater salary and/or bonus payments than non-POIs.
* Similarly, POIs may have been rewarded with increased stock options, which would result in them having greater total stock values. We might also consider that individuals with greater stock positions to begin with would be interested in maximising the value of the company, potentially by colluding to artificially inflate the company’s profits as was the case in the Enron scandal.

To select which of these features to use in my classifier, I tried fitting them all to a decision tree in scikit-learn, and used the ```.feature_importances_``` attribute to determine which of the features were most influential, which reported the following (averaged over 1000 iterations):

```
total_payments: 0.00613240418118
total_stock_value: 0.164538016426
from_poi_to_this_person: 0.270466785336
from_this_person_to_poi: 0.0402838269454
to_messages: 0.0376488095238
from_messages: 0.0240217770035
from_poi_rate: 0.171078863412
to_poi_rate: 0.285829517173
```

Given the relatively similar levels of importance attributed to 'total_stock_value', ‘from_poi_to_this_person', 'from_poi_rate' and 'to_poi_rate', I initially chose to use all four of these features in my final analysis. After visualising the data for these four features, I also chose to remove a further three outliers: Gene Humphrey, John J Lavorato and Jeffrey M Donahue Jr (none of whom were POIs).

After tuning my algorithm (addressed below) however, I tried removing each of these features in turn and discovered that performance improved when 'from_poi_rate' was excluded. Therefore I decided to remove it from my final classifier. Please note that the results presented in the ‘Algorithm selection and tuning’ section include my original four features, rather than the final three.

## Algorithm selection and tuning

*What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?*

I tried three different algorithms: Naïve Bayes (“NB”), decision trees (“DT”) and random forest (“RF”). Using each classifier’s default values, I observed the following performance using ‘tester.py’:
```
NB: Accuracy: 0.87107, Precision: 0.61042, Recall: 0.26950
DT: Accuracy: 0.79850, Precision: 0.27408, Recall: 0.24900
RF: Accuracy: 0.84500, Precision: 0.40319, Recall: 0.17700
```
For out-of-the-box performance, NB appears to be the strongest overall, while DT was the most balanced. 

*What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm?*

While the NB algorithm performed strongly in my initial testing, it has less scope for optimisation or ‘tuning’ than my other two algorithms. Tuning an algorithm ensures that the parameters used reflect the problem being investigated, thereby obtaining maximum performance (i.e. accuracy). They can also guard against over-fitting, e.g. by making sure a DT doesn’t create separate branches for every individual outcome.

I tuned the DT and RF algorithms using scikit-learn’s GridSearchCV function, which tested different combinations of 'criterion', 'min_samples_split', 'max_features', 'max_depth' and (for RF) 'n_estimators'. The best results I was able to achieve were as follows:

```
DT: Accuracy: 0.82521, Precision: 0.36799, Recall: 0.31150
RF: Accuracy: 0.84179, Precision: 0.40236, Recall: 0.22150
```

Based on these results, I decided to use DT in my final analysis, as it provided the most balanced performance between evaluation metrics.

## Validation

*What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?*

Validation is the process of retaining a sample of the data set and using it to test the classifier once it has been tuned. This is important, because if the classifier is only tuned using a training and test set, it may become overfitted to the test set and thus underperform in real life applications. The validation set therefore acts as a final check to ensure overfitting has not occurred. 

With small data sets such as the Enron set, the data sampling process that creates the training, test and validation sets can have a significant impact on the classifier’s performance – for example, if the distribution of data in the training set do not reflect that of the wider set. To overcome this, I used a cross-validation function (stratified shuffle split), which randomly splits the data into k samples and trains the classifier on each of the k-1 samples, before validating it on the remaining data. The classifier’s performance is thus averaged across each of the samples.

## Evaluation

*Give at least 2 evaluation metrics and your average performance for each of them. Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.*

I evaluated my classifier primarily using the ‘precision’ and ‘recall’ metrics. [Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall) defines these as measures of “exactness” and “completeness”, respectively. In the context of this data set, precision measures the proportion of correctly identified POIs out of the total number of individuals identified as POI by our classifier. Recall measures the proportion of correctly identified POIs out of the total number of POIs in the data set.

Using the ‘tester.py’ script, averaged over 10 iterations, my classifier achieved the following performance:

```
Precision: 0.372954, Recall: 0.32205
```
Therefore, approximately 37.3% of the individuals identified as POIs by the classifier were, in fact, POIs, which corresponded to 32.2% of the overall number of POIs in the data set. Clearly this is not particularly accurate, which would impact its usefulness on any similar investigations in future. 

The features I developed during this investigation ('from_poi_rate' and 'to_poi_rate') improved the performance of my classifier, as the following results demonstrate:
```
Neither feature used: Precision: 0.26231, Recall: 0.19450
'from_poi_rate' used: Precision: 0.24315, Recall: 0.22200
'to_poi_rate' used: Precision: 0.37486, Recall: 0.33250
Both features used: Precision: 0.36337, Recall: 0.31050
```
(note 'total_stock_value' and 'from_poi_to_this_person' are used in each test, but the classifier parameters remain constant)

It is possible that other potential features lie in the data set that would lead to more accurate classifications. For example, we might search all the emails for the presence of keywords, e.g. “[special purpose entities]( https://en.wikipedia.org/wiki/Enron_scandal#Special_purpose_entities)” and determine the frequency that they were used by individuals. I would be interested in exploring this further in a future investigation.

## Declaration

I hereby confirm that this submission is my work. I have cited above the origins of any parts of the submission that were taken from Websites, books, forums, blog posts, github repositories, etc.

## List of resources used

* https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html#numpy.hstack  
* https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.isnan.html  
* http://scikit-learn.org/stable/modules/feature_selection.html  
* http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
* http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
* http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html  
* http://scikit-learn.org/stable/modules/cross_validation.html
