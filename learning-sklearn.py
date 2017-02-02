import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import cross_validation

testData = pd.read_csv("D:/courses/CS460G/hw5_data/test.csv")
train = pd.read_csv("D:/courses/CS460G/hw5_data/train.csv")
trainLab = train.ix[:,0]
trainData = train.ix[:,1:]

X = trainData.values
y = trainLab.values

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
##########################################################This is Round 1
##cross validation for number of estimators in random forest
tuning_vec = list(range(10,30))
Avg_TPRvec=[]
for tuning in tuning_vec:
	k_fold = cross_validation.KFold(n=42000,n_folds=5,shuffle=True)
	clf = RandomForestClassifier(n_estimators=tuning)
	True_score=[]
	for train,test in k_fold:
		Tree_clf = clf.fit(X[train], y[train])
		y_pred_test = Tree_clf.predict(X[test,])
		y_true = y[test]
		True_score = np.append(True_score,sum(y_pred_test == y_true)/len(y_true))
	Avg_TPR = np.mean(True_score)
	Avg_TPRvec = np.append(Avg_TPRvec,Avg_TPR)
	
	
plt.xlable("number of trees")
plt.ylable("AvgTPR")
plt.plot(tuning_vec,Avg_TPRvec)
plt.show()


ind=np.where(Avg_TPRvec==max(Avg_TPRvec))
tuning = tuning_vec[15]
#tuning =25
clf = RandomForestClassifier(n_estimators=25)
RF_clf = clf.fit(X,y)
y_pred_test = RF_clf.predict(testData)
test_labels=[]
for i in range(len(y_pred_test)):
	test_labels.append([i+1,y_pred_test[i]])
np.savetxt('testLab.csv',test_labels,delimiter=',',fmt='%f')
#####################################################


#############################################################This is round 2
from sklearn.neighbors import KNeighborsClassifier
##cross validation for number of Neighbors
##randomly pick 10000 of the training samples
indi=np.random.random_inegers(low=0, high=42000, size=10000)
RnX = X[indi,:]
Rny = y[indi]
N_vec = list(range(3,10))
Avg_TPRvec=[]
for N_value in N_vec:
	k_fold = cross_validation.KFold(n=10000,n_folds=5,shuffle=True)
	clf = KNeighborsClassifier(n_neighbors=N_value)
	True_score=[]
	for train,test in k_fold:
		KNN_clf = clf.fit(RnX[train],Rny[train])
		Rny_pred_test = KNN_clf.predict(Rny[test,])
		Rny_true = Rny[test]
		True_score = np.append(True_score, sum(Rny_pred_test == Rny_true)/len(Rny_true))
	Avg_TPR = np.mean(True_score)
	Avg_TPRvec = np.append(Avg_TPRvec,Avg_TPR)

clf = KNeighborsClassifier(n_neighbors=3)
KNN_clf = clf.fit(X,y)
y_pred_test1 = KNN_clf.predict(testData[0:14000])
y_pred_test2 = KNN_clf.predict(testData[14000:])
y_pred_test = np.append(y_pred_test1,y_pred_test2)
test_labels=[]
for i in range(len(y_pred_test)):
	test_labels.append([i+1,y_pred_test[i]])
np.savetxt('D:/courses/CS460G/testLabround2.csv',test_labels,delimiter=',',fmt='%f')
######################################
#######################################
from sklearn import linear_model
clf = linear_model.SGDClassifier(alpha=0.0001,loss="log",penalty="l2")
SGD_clf = clf.fit(X,y)
y_pred_train = SGD_clf.predict(X)
y_pred_test = SGD_clf.predict(testData)
test_labels=[]
for i in range(len(y_pred_test)):
	test_labels.append([i+1,y_pred_test[i]])
np.savetxt('D/courses/CS460G/testLabround3.CSV',test_labels,delimiter=',',fmt='%f')
	
	






















