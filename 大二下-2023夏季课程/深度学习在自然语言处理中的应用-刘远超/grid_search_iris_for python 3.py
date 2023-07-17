# -*- coding: utf-8 -*-  
"""
============================================================
Parameter estimation using grid search with cross-validation
============================================================
"""

from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
#不显示warning
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

print(__doc__)

# Loading the Digits dataset
iris = datasets.load_iris()

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'],                     'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']


for score in scores:
    print("# Tuning hyper-parameters for %s" % score) 
    print ('%s_weighted' % score)
    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_weighted' % score) #1/5作为验证集
    clf.fit(X_train, y_train)#用前一半train数据再做5折交叉验证，因为之前train_test_split已经分割为2份了
    print("Best parameters set found on development set:"  )  
    print(clf.best_params_)    
    print("Grid scores on development set:")    
    
#    for params, mean_score, scores in clf.grid_scores_:
#        print("%0.3f (+/-%0.03f) for %r"
#              % (mean_score, scores.std() * 1.96, params))
    #新版本
    means = clf.cv_results_['mean_test_score']
    params = clf.cv_results_['params']
    for mean,param in zip(means,params):
        print("%f  with:   %r" % (mean,param))
 

    print("Detailed classification report:")    
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")    
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    
# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.