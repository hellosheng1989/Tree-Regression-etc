import time
import pylab
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from itertools import combinations
plt.ion()

test = pd.read_csv('D:\courses\CS460G\hw2_data\data\hw2.test')
train = pd.read_csv('D:\courses\CS460G\hw2_data\data\hw2.train')
testFea = test.values[:,0]
testLab = test.values[:,1]
trainFea = train.values[:,0]
trainLab = train.values[:,1]
def fit_lr_normal(data=testFea,labels=testLab,k=3):
    #normalization of testFea
    x = (data - np.mean(data))/(max(data)-min(data))
    y = np.matrix(labels).T
    tempX =  np.array([1]*len(x))
    for i in range(k):
        tempX = np.column_stack((tempX,x**(i+1)))
    X = np.matrix(tempX)
    Y = np.matrix(labels).T
    if k!=0: 
        theta = np.dot(np.dot(X.T,X).I,np.dot(X.T,Y))
        predy = X*theta
        pltx = list(x)
        plty = predy.tolist()
        truey = y.tolist()
        plt.title("regression model")
        plt.scatter(pltx,plty)
        plt.show()
        plt.title("true testing data")
        plt.scatter(pltx,truey)
        plt.show()  
    else:
        theta = np.mean(Y)
        pltx = list(x)
        truey = y.tolist()
        plt.axhline(y=theta,hold=None)
        plt.scatter(pltx,truey)
        plt.title("k=0,put model and testing data together")
        plt.show()    
    return theta,max(data),min(data),np.mean(data)
    

def fit_lr_gd(data=testFea,labels=testLab,k=4,eps=0.0001,rate=0.03):
    #normalization of testFea
    x = (data - np.mean(data))/(max(data)-min(data))
    y = np.matrix(labels).T
    tempX =  np.array([1]*len(x))
    for i in range(k):
        tempX = np.column_stack((tempX,x**(i+1)))
    X = np.matrix(tempX)
    if k!=0:
        theta1 = np.matrix([np.mean(labels)]+[0]*k).T    # intial theta to be (mean(y),k 0's) if k>=1
        Deri = (np.dot(X.T,X)*theta1 - X.T*y)*2/len(x)
        theta2 = theta1 - rate*Deri                      # one iteration to have two different theta, for starting loops
        while sum(abs(theta1-theta2))>eps:
            theta1 = theta2
            Deri = (np.dot(X.T,X)*theta1 - X.T*y)*2/len(x)
            theta2 = theta2 - rate*Deri
        predy = X*theta2
        pltx = list(x)
        plty = predy.tolist()
        truey = y.tolist()
        plt.title("regression model")
        plt.scatter(pltx,plty)
        plt.show()
        plt.title("true testing data")
        plt.scatter(pltx,truey)
        plt.show() 
    else:
        theta2 = np.mean(y)
        pltx = list(x)
        truey = y.tolist()
        plt.axhline(y=theta2,hold=None)
        plt.scatter(pltx,truey)
        plt.title("k=0,put model and testing data together")
        plt.show()        
    return theta2,max(data),min(data),np.mean(data)    
    

# k is the order of regression = len(theta)-1, mini, maxi, mean is from previous fitting for normalization of predictors
# model = theta ,which is a matrix of parameters/theta's
results = fit_lr_normal(data=testFea,labels=testLab,k=4)           # here k can be changed to 1,2,3,4
results = fit_lr_gd(data=testFea,labels=testLab,k=4)

theta = results[0]
maxi = results[1]
mini = results[2]
mean = results[3]
def predict_lr(model=theta,data=testFea,k=4,mini=mini,maxi=maxi,mean=mean):
    x = (data - mean)/(maxi - mini)
    tempX =  np.array([1]*len(x))
    for i in range(k):
        tempX = np.column_stack((tempX,x**(i+1)))
    preX = np.matrix(tempX)
    preY = preX*model
    preY = preY.reshape((len(testFea),1))
    return preY

preY = predict_lr(model=theta,data=testFea,k=4,mini=mini,maxi=maxi,mean=mean)  
#compute MSE   
def compute_mse(labels_ground_truth=testLab,labels_estimated=preY):
    labels_ground_truth = labels_ground_truth.reshape((len(labels_ground_truth),1))
    labels_estimated = labels_estimated.reshape((labels_estimated.shape[0],1))
    ss = labels_ground_truth - labels_estimated
    mse = ss.T*ss / len(labels_ground_truth)
    return mse

mse = compute_mse(labels_ground_truth=testLab,labels_estimated=preY)    


###plots for generative model
def fit_lr_guess(data=testFea,labels=testLab,k=2):
    #normalization of testFea
    x = data
    y = np.matrix(labels).T
    tempX =  np.array([1]*len(x))
    for i in range(k):
        tempX = np.column_stack((tempX,x**(i+1)))
    X = np.matrix(tempX)
    Y = np.matrix(labels).T
    theta = np.dot(np.dot(X.T,X).I,np.dot(X.T,Y))
    predy = X*theta
    pltx = list(x)
    plty = predy.tolist()
    truey = y.tolist()
    plt.plot(pltx,plty)
    plt.scatter(pltx,truey)
    plt.show()  
 
#use more data for estimation 
Fea = np.append(testFea,trainFea)
Lab = np.append(testLab,trainLab)
results = fit_lr_normal(data=Fea,labels=Lab,k=3)

#results
#(matrix([[  0.93833708],
#         [  1.56056127],
#         [ 35.47343776],
#         [-46.21583462]]), 1.4255, -1.4466000000000001, -0.047456904878048795

guessofNoise = np.std(Lab-1-Fea-Fea**2-Fea**3
# 3.636324256266141






#Apply Regression Models to the Housing Data 
from sklearn.cross_validation import KFold
housingdata = pd.read_csv('D:\courses\CS460G\hw2_data\data\housing.data',delim_whitespace=True)
housingFea = housingdata.values[:,0:12]
housingLab = housingdata.values[:,13]

def compute_err(TrainFea=housingFea,TrainLab=housingLab,TestFea=housingFea[0:10],TestLab=housingLab[0:10]):
    X = np.matrix(TrainFea)
    Y = np.matrix(TrainLab).T
    if np.linalg.det(np.dot(X.T,X))==0:
        return 10000000000000
    else:
        theta = np.dot(np.dot(X.T,X).I,np.dot(X.T,Y))
        PreY = TestFea*theta
        PreY = np.array(PreY)
        TestLab = np.array(TestLab).reshape((len(TestLab),1))
        msePre = sum(TestLab - PreY)**2
        return msePre    
    
    
one = np.array([1]*len(housingLab)).reshape((len(housingLab),1)) 
a = housingFea
b = housingFea**2 
c = housingFea**3
F = np.column_stack((one,a,b,c))

#try all subsets of features to get the best model
D = F.shape[1]
min_error = 1000000000000000
errlist=[]
for i in range(D):
    for j in range(i+1,D):
        for n in range(j+1,D):
            err=0    
            kf = KFold(99, n_folds=5)
            lkf = list(kf)
            for k in range(5):
                TrainInd = lkf[k][0]
                TestInd  = lkf[k][1]
                TrainData = F[TrainInd,:]
                TestData = F[TestInd,:]
                msePreErr = compute_err(TrainFea=TrainData[:,[i,j,n]],TrainLab=housingLab[TrainInd],TestFea=TestData[:,[i,j,n]],TestLab=housingLab[TestInd])
                err = err + msePreErr
            if err<min_error:
                bestModel = [i,j,n]
                min_error = err
            errlist = errlist + [[err,[i,j,n]]]
errlist = np.array(errlist)
#find five models with minimum MSPE
TempErrlist = errlist
Model = []
for i in range(5):
    errInd = np.where(TempErrlist[:,0]==min(TempErrlist[:,0]))
    Model = np.append(Model,TempErrlist[errInd,])
    TempErrlist = np.delete(TempErrlist,(errInd),axis=0)

Model  # model will show us the mean squared prediction error and corresponded index

#Bonus: fit a model with kth polynomial and regularization parameter lambda
def fit_lr_Tikhonov(data=testFea,labels=testLab,k=4,lam=10):
    #normalization of testFea
    x = (data - np.mean(data))/(max(data)-min(data))
    y = np.matrix(labels).T
    tempX =  np.array([1]*len(x))
    for i in range(k):
        tempX = np.column_stack((tempX,x**(i+1)))
    X = np.matrix(tempX)
    Y = np.matrix(labels).T
    theta = (np.dot(X.T,X) + np.diag([lam]*X.shape[1])).I * np.dot(X.T,Y)    # (X^t*X + lambda*I)^-1X*Y
    predy = X*theta
    pltx = list(x)
    plty = predy.tolist()
    truey = y.tolist()
    plt.plot(pltx,plty)
    plt.scatter(pltx,truey)
    plt.show()
    return theta,max(data),min(data),np.mean(data)











































    
    
    