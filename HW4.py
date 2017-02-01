# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 19:55:55 2015

@author: SSL
"""

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation
from sklearn import linear_model 
import pylab as pl

testdata = pd.read_csv('D:\courses\CS460G\hw4_data\caltechTestData.dat',header=None,sep=' ')
traindata = pd.read_csv('D:\courses\CS460G\hw4_data\caltechTrainData.dat',header=None,sep=' ')
trainLabel = pd.read_csv('D:\courses\CS460G\hw4_data\caltechTrainLabel.dat',header=None,sep=' ')


X = traindata.values[:,0:2700]
y = trainLabel.values.reshape((1440,))
testdata_values = testdata.values[:,0:2700]

#too many features which make the computation too slow, I need to do dimension reduction at first.
from sklearn.decomposition import PCA
pca=PCA(n_components=50)
pca_obj=pca.fit(X)
ExplainedVar = sum(pca_obj.explained_variance_ratio_)   #about 87% of the total variance has been explained
NewX = pca_obj.transform(X)
NewTestData = pca_obj.transform(testdata_values)



#cross validation for choose tuning parameter C, C controls the penaty for the loss function
Cvec = [1e-2,5e-2,1e-1,5e-1,1,5,10]
Avg_TPRvec=[]
for Cvalue in Cvec:
    k_fold = cross_validation.KFold(n=1440, n_folds=5,shuffle=True)
    logreg = linear_model.LogisticRegression(C=Cvalue)
    True_score=[]
    for train,test in k_fold:
        Multi_Classification =  logreg.fit(NewX[train], y[train])
        y_pred_test=Multi_Classification.predict(NewX[test,])
        y_true = y[test]
        True_score = np.append(True_score,sum(y_pred_test == y_true)/len(y_true))
    Avg_TPR = np.mean(True_score)
    Avg_TPRvec = np.append(Avg_TPRvec,Avg_TPR)


##use the best C value to get the TPR for training data and do the prediction
Cvec = np.array(Cvec)
C_value = Cvec[np.where(Avg_TPRvec==max(Avg_TPRvec))[0]]     ##C_value = 0.05
logreg = linear_model.LogisticRegression(C=C_value)
log_Classification =  logreg.fit(NewX, y)
y_pred_train = log_Classification.predict(NewX)
y_true_train = y  
TPR = sum(y_pred_train == y_true_train)/len(y_true_train)   ##TPR = 62.986%
#prediction for test data
y_pred_test = log_Classification.predict(NewTestData)
np.savetxt('testLabel.dat',y_pred_test)

#confusion matrix
Con_Mat = confusion_matrix(y_true_train, y_pred_train)
pl.matshow(Con_Mat)
pl.title('Confusion matrix(classification)')
pl.xlabel('predicted class')
pl.ylabel('true class')
pl.colorbar()
pl.show()