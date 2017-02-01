# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:21:58 2015

@author: SSL
"""


import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import scipy.optimize
syn1 = pd.read_csv('D:\courses\CS460G\hw3_data\data\synthetic-1.csv')
syn2 = pd.read_csv('D:\courses\CS460G\hw3_data\data\synthetic-2.csv')
syn3 = pd.read_csv('D:\courses\CS460G\hw3_data\data\synthetic-3.csv')
syn4 = pd.read_csv('D:\courses\CS460G\hw3_data\data\synthetic-4.csv')
syn5 = pd.read_csv('D:\courses\CS460G\hw3_data\data\synthetic-5.csv')
syn6 = pd.read_csv('D:\courses\CS460G\hw3_data\data\synthetic-6.csv')
Fea1 = syn1.values[:,0:2]
Lab1 = syn1.values[:,2]
Fea2 = syn2.values[:,0:2]
Lab2 = syn2.values[:,2]
Fea3 = syn3.values[:,0:2]
Lab3 = syn3.values[:,2]
Fea4 = syn4.values[:,0:2]
Lab4 = syn4.values[:,2]
Fea5 = syn5.values[:,0:2]
Lab5 = syn5.values[:,2]
Fea6 = syn6.values[:,0:2]
Lab6 = syn6.values[:,2]


def g(z):
    g = 1/(1+np.exp(-z))
    return g

#theta should be a matrix with one row, x shoule be a matrix with one column, they have the same dimensions.
def h(theta,x):
    h = g(theta*x)
    return h

def deri_h(theta,x):
    d = theta.shape[1]
    Grad = []
    for i in range(d):
        tempGrad = x[i]/(1+np.exp(-theta*x))**2
        Grad = np.append(Grad,tempGrad)
    return Grad
#create a set of parameters, thetaLB is a list/array of lowerbound of theta's
# thetaUB is a list/array of upper bound of theta's, size is number of theta 
# for each dimension
def sampling_grid(thetaLB,thetaUB,size):
    d = len(thetaLB)
    gd = []
    for i in range(d):
        temp = np.linspace(thetaLB[i],thetaUB[i],size)
        gd = np.append(gd,temp)
    thetaAll = gd.reshape((d,size))
    thetaAll = thetaAll.tolist()
    list = thetaAll
    result = itertools.product(*list)
    griddata = []
    for i in result:
        griddata.append(i)
    GridData = np.array(griddata)    
    return GridData

# create two function one for calculating J(theta), one for J'(theta)
# theta should be a 1*d matrix
# Features is np.array , Labels is np.array
def J(theta,Features=Fea1,Labels=Lab1,penal=0.1):
    m = len(Labels)
    theta = np.matrix(theta)
    TempJ = 0
    for i in range(m):
        tempx = np.matrix(Features[i]).T
        TempJ = (TempJ + (Labels[i]*np.log(h(theta,tempx))+(1-Labels[i])*np.log(1-h(theta,tempx)))) * (-1/m)
    J = np.array(TempJ) + penal*(theta*theta.T)
    return J 


#logistic regression
def logi_reg_gd(Features=Fea1,Labels=Lab1,initialTheta=np.matrix([1,-1]),penal=0):
    theta = scipy.optimize.fmin_bfgs(J,x0=initialTheta,args=(Features,Labels,penal))
    return theta

def Get_err_gd(Features=Fea1,Labels=Lab1,initialTheta=np.matrix([2,-1]),penal=0):
    Parameters = np.matrix(logi_reg_gd(Features=Features,Labels=Labels,initialTheta=initialTheta,penal=penal))
    n = len(Labels)
    misNum = 0
    for i in range(n):
        TempX = np.matrix(Features[i]).T
        tempProb = 1/(1+np.exp(-Parameters*TempX))
        if (tempProb>0.5 and Labels[i]==0) or (tempProb<0.5 and Labels[i]==1):
            misNum = misNum + 1
    err = misNum/n
    return err
##plot
## x is d*N matrix, Parameters is 1*d matrix, return 1*N matrix with Probabilities for each test points
def Prob(Parameters,x):
    Proba = 1/(1+np.exp(-Parameters*x))
    return Proba

def plotDB_rawFea_gd(Fea=Fea1,Lab=Lab1,initialTheta=np.array([2,-1]),penal=0):
# create the domain for the plot
    x_min = min(Fea[:,0])-0.5; x_max = max(Fea[:,0])+0.5
    y_min = min(Fea[:,1])-0.5; y_max = max(Fea[:,1])+0.5

    x1 = np.linspace(x_min, x_max, 200)
    y1 = np.linspace(y_min, y_max , 200)
    x,y = np.meshgrid(x1, y1)

#make 2 x N matrix of the sample points (x,y)
    data1 = np.vstack((x.ravel(),y.ravel()))
    Parameters1 = logi_reg_gd(Features=Fea,Labels=Lab,initialTheta=initialTheta,penal=penal)
    Parameters1 = np.matrix(Parameters1)
    z1 = Prob(Parameters=Parameters1,x=data1)
    z1 = z1.reshape(x.shape)

# Make the plots
# show the function value in the background
    cs = plt.imshow(z1,
        extent=(x_min,x_max,y_max,y_min), # define limits of grid, note reversed y axis
        cmap=plt.cm.jet)
    plt.clim(0,1) # defines the value to assign the min/max color

# draw the line on top
    levels = np.array([.5])
    cs_line = plt.contour(x,y,z1,levels)

# add a color bar
    CB = plt.colorbar(cs)

# add data points
    for lab,x1t,x2t in zip(Lab,Fea[:,0],Fea[:,1]):
        if lab==1:
            plt.scatter(x1t,x2t,marker='v')
        else:
            plt.scatter(x1t,x2t,marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('visualization of classifier,implimentation #1,raw features') 
    plt.show()


plotDB_rawFea_gd(Fea=Fea1,Lab=Lab1,initialTheta=np.array([2,-1]),penal=0)
Get_err_gd(Features=Fea1,Labels=Lab1,initialTheta=np.matrix([2,-1]),penal=0)

plotDB_rawFea_gd(Fea=Fea2,Lab=Lab2,initialTheta=np.array([2,-1]),penal=0)
Get_err_gd(Features=Fea2,Labels=Lab2,initialTheta=np.matrix([2,-1]),penal=0)

plotDB_rawFea_gd(Fea=Fea3,Lab=Lab3,initialTheta=np.array([2,-1]),penal=0)
Get_err_gd(Features=Fea3,Labels=Lab3,initialTheta=np.matrix([2,-1]),penal=0)

plotDB_rawFea_gd(Fea=Fea4,Lab=Lab4,initialTheta=np.array([2,1]),penal=0)
Get_err_gd(Features=Fea4,Labels=Lab4,initialTheta=np.matrix([2,1]),penal=0)

plotDB_rawFea_gd(Fea=Fea5,Lab=Lab5,initialTheta=np.array([2,-1]),penal=0)
Get_err_gd(Features=Fea5,Labels=Lab5,initialTheta=np.matrix([2,-1]),penal=0)

plotDB_rawFea_gd(Fea=Fea6,Lab=Lab6,initialTheta=np.array([2,-1]),penal=0)
Get_err_gd(Features=Fea6,Labels=Lab6,initialTheta=np.matrix([2,-1]),penal=0)


#plot for quadratic features

# create quadratic features
QFea1 = np.vstack((
np.ones(Fea1[:,0].size), # add the bias term
Fea1[:,0],Fea1[:,1],Fea1[:,0]*Fea1[:,1],
Fea1[:,0]**2,Fea1[:,1]**2)) 
QFea1 = QFea1.T

QFea2 = np.vstack((
np.ones(Fea2[:,0].size), # add the bias term
Fea2[:,0],Fea2[:,1], Fea2[:,0]*Fea2[:,1],
Fea2[:,0]**2,Fea2[:,1]**2)) 
QFea2 = QFea2.T

QFea3 = np.vstack((
np.ones(Fea3[:,0].size), # add the bias term
Fea3[:,0],Fea3[:,1], Fea3[:,0]*Fea3[:,1],
Fea3[:,0]**2,Fea3[:,1]**2)) 
QFea3 = QFea3.T

QFea4 = np.vstack((
np.ones(Fea4[:,0].size), # add the bias term
Fea4[:,0],Fea4[:,1], Fea4[:,0]*Fea4[:,1],
Fea4[:,0]**2,Fea4[:,1]**2)) 
QFea4 = QFea4.T

QFea5 = np.vstack((
np.ones(Fea5[:,0].size), # add the bias term
Fea5[:,0],Fea5[:,1], Fea5[:,0]*Fea5[:,1],
Fea5[:,0]**2,Fea5[:,1]**2)) 
QFea5 = QFea5.T

QFea6 = np.vstack((
np.ones(Fea6[:,0].size), # add the bias term
Fea6[:,0],Fea6[:,1], Fea6[:,0]*Fea6[:,1],
Fea6[:,0]**2,Fea6[:,1]**2)) 
QFea6 = QFea6.T


def plotDB_QuadraticFea_gd(Fea=QFea1,Lab=Lab1,initialTheta=np.matrix([-0.5,0.23,-0.5,0.15,0.15,-0.5]),penal=0):
# create the domain for the plot
    x_min = min(Fea[:,1])-0.5; x_max = max(Fea[:,1])+0.5
    y_min = min(Fea[:,2])-0.5; y_max = max(Fea[:,2])+0.5

    x1 = np.linspace(x_min, x_max, 200)
    y1 = np.linspace(y_min, y_max , 200)
    x,y = np.meshgrid(x1, y1)
    
# make a 3 x N matrix of the sample points
    data1 = np.vstack((
    np.ones(x.size), # add the bias term
    x.ravel(), # make the matrix into a vector
    y.ravel(), 
    x.ravel()*y.ravel(),
    x.ravel()**2,
    y.ravel()**2)) # add a quadratic term for fun


    Parameters1 = logi_reg_gd(Features=Fea,Labels=Lab,initialTheta=initialTheta,penal=penal)
    Parameters1 = np.matrix(Parameters1)
    z1 = Prob(Parameters=Parameters1,x=data1)
    z1 = z1.reshape(x.shape)

# Make the plots
# show the function value in the background
    cs = plt.imshow(z1,
        extent=(x_min,x_max,y_max,y_min), # define limits of grid, note reversed y axis
        cmap=plt.cm.jet)
    plt.clim(0,1) # defines the value to assign the min/max color

# draw the line on top
    levels = np.array([.5])
    cs_line = plt.contour(x,y,z1,levels)

# add a color bar
    CB = plt.colorbar(cs)

# add data points
    for lab,x1t,x2t in zip(Lab,Fea[:,1],Fea[:,2]):
        if lab==1:
            plt.scatter(x1t,x2t,marker='v')
        else:
            plt.scatter(x1t,x2t,marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('visualization of classifier,implimentataion #1, Quadratic') 
    plt.show()


plotDB_QuadraticFea_gd(Fea=QFea1,Lab=Lab1,initialTheta=np.matrix([-0.5,0.23,-0.5,0.15,0.15,-0.5]),penal=0)
Get_err_gd(Features=QFea1,Labels=Lab1,initialTheta=np.matrix([-0.5,0.23,-0.5,0.15,0.15,-0.5]),penal=0)

plotDB_QuadraticFea_gd(Fea=QFea2,Lab=Lab2,initialTheta=np.matrix([2,0.2,0.8,1.5,-0.1,-1.26]),penal=0)
Get_err_gd(Features=QFea2,Labels=Lab2,initialTheta=np.matrix([2,0.2,0.8,1.5,-0.1,-1.26]),penal=0)

plotDB_QuadraticFea_gd(Fea=QFea3,Lab=Lab3,initialTheta=np.matrix([1.78,0.25,0.4,0.9,-1.2,0.05]),penal=0)
Get_err_gd(Features=QFea3,Labels=Lab3,initialTheta=np.matrix([1.78,0.25,0.4,0.9,-1.2,0.05]),penal=0)

plotDB_QuadraticFea_gd(Fea=QFea4,Lab=Lab4,initialTheta=np.matrix([1.8,1.2,0.5,0,-0.1,-0.15]),penal=0)
Get_err_gd(Features=QFea4,Labels=Lab4,initialTheta=np.matrix([1.8,1.2,0.5,0,-0.1,-0.15]),penal=0)

plotDB_QuadraticFea_gd(Fea=QFea5,Lab=Lab5,initialTheta=np.matrix([0.3,0.1,-0.3,1.9,-0.2,0]),penal=0)
Get_err_gd(Features=QFea5,Labels=Lab5,initialTheta=np.matrix([0.3,0.1,-0.3,1.9,-0.2,0]),penal=0)

plotDB_QuadraticFea_gd(Fea=QFea6,Lab=Lab6,initialTheta=np.matrix([0,0,-0.3,0,0,0]),penal=0)
Get_err_gd(Features=QFea6,Labels=Lab6,initialTheta=np.matrix([0,0,-0.3,0,0,0]),penal=0)














#implementation #2: High-level Logistic Regression Function
from sklearn import linear_model
classfier1 = linear_model.LogisticRegression(penalty = 'l2')
classfier2 = linear_model.LogisticRegression(penalty = 'l2')
classfier3 = linear_model.LogisticRegression(penalty = 'l2')
classfier4 = linear_model.LogisticRegression(penalty = 'l2')
classfier5 = linear_model.LogisticRegression(penalty = 'l2')
classfier6 = linear_model.LogisticRegression(penalty = 'l2')
a1=classfier1.fit(Fea1,Lab1)
a2=classfier2.fit(Fea2,Lab2)
a3=classfier3.fit(Fea3,Lab3)
a4=classfier4.fit(Fea4,Lab4)
a5=classfier5.fit(Fea5,Lab5)
a6=classfier6.fit(Fea6,Lab6)

#get misclassification error
def Get_err(Features=Fea1,Labels=Lab1,LogiObj=a1):
    Parameters = np.matrix(LogiObj.coef_)
    n = len(Labels)
    misNum = 0
    for i in range(n):
        TempX = np.matrix(Features[i]).T
        tempProb = 1/(1+np.exp(-Parameters*TempX))
        if (tempProb>0.5 and Labels[i]==0) or (tempProb<0.5 and Labels[i]==1):
            misNum = misNum + 1
    err = misNum/n
    return err



##plot
## x is d*N matrix, Parameters is 1*d matrix, return 1*N matrix with Probabilities for each test points
def Prob(Parameters,x):
    Proba = 1/(1+np.exp(-Parameters*x))
    return Proba


def plotDB_rawFea(Fea=Fea1,Lab=Lab1,LogiObj = a1):
# create the domain for the plot
    x_min = min(Fea[:,0])-0.5; x_max = max(Fea[:,0])+0.5
    y_min = min(Fea[:,1])-0.5; y_max = max(Fea[:,1])+0.5

    x1 = np.linspace(x_min, x_max, 200)
    y1 = np.linspace(y_min, y_max , 200)
    x,y = np.meshgrid(x1, y1)

#make 2 x N matrix of the sample points (x,y)
    data1 = np.vstack((x.ravel(),y.ravel()))
    Parameters1 = np.matrix(LogiObj.coef_)
    z1 = Prob(Parameters=Parameters1,x=data1)
    z1 = z1.reshape(x.shape)

# Make the plots
# show the function value in the background
    cs = plt.imshow(z1,
        extent=(x_min,x_max,y_max,y_min), # define limits of grid, note reversed y axis
        cmap=plt.cm.jet)
    plt.clim(0,1) # defines the value to assign the min/max color

# draw the line on top
    levels = np.array([.5])
    cs_line = plt.contour(x,y,z1,levels)

# add a color bar
    CB = plt.colorbar(cs)

# add data points
    for lab,x1t,x2t in zip(Lab,Fea[:,0],Fea[:,1]):
        if lab==1:
            plt.scatter(x1t,x2t,marker='v')
        else:
            plt.scatter(x1t,x2t,marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('visualization of classifier,implimentation #2,raw Features') 
    plt.show()


plotDB_rawFea(Fea=Fea1,Lab=Lab1,LogiObj = a1)
Get_err(Features=Fea1,Labels=Lab1,LogiObj=a1)

plotDB_rawFea(Fea=Fea2,Lab=Lab2,LogiObj = a2)
Get_err(Features=Fea2,Labels=Lab2,LogiObj=a2)

plotDB_rawFea(Fea=Fea3,Lab=Lab3,LogiObj = a3)
Get_err(Features=Fea3,Labels=Lab3,LogiObj=a3)

plotDB_rawFea(Fea=Fea4,Lab=Lab4,LogiObj = a4)
Get_err(Features=Fea4,Labels=Lab4,LogiObj=a4)

plotDB_rawFea(Fea=Fea5,Lab=Lab5,LogiObj = a5)
Get_err(Features=Fea5,Labels=Lab5,LogiObj=a5)

plotDB_rawFea(Fea=Fea6,Lab=Lab6,LogiObj = a6)
Get_err(Features=Fea6,Labels=Lab6,LogiObj=a6)


#plot for quadratic features

# create quadratic features
QFea1 = np.vstack((
np.ones(Fea1[:,0].size), # add the bias term
Fea1[:,0],Fea1[:,1],Fea1[:,0]*Fea1[:,1],
Fea1[:,0]**2,Fea1[:,1]**2)) 
QFea1 = QFea1.T

QFea2 = np.vstack((
np.ones(Fea2[:,0].size), # add the bias term
Fea2[:,0],Fea2[:,1], Fea2[:,0]*Fea2[:,1],
Fea2[:,0]**2,Fea2[:,1]**2)) 
QFea2 = QFea2.T

QFea3 = np.vstack((
np.ones(Fea3[:,0].size), # add the bias term
Fea3[:,0],Fea3[:,1], Fea3[:,0]*Fea3[:,1],
Fea3[:,0]**2,Fea3[:,1]**2)) 
QFea3 = QFea3.T

QFea4 = np.vstack((
np.ones(Fea4[:,0].size), # add the bias term
Fea4[:,0],Fea4[:,1], Fea4[:,0]*Fea4[:,1],
Fea4[:,0]**2,Fea4[:,1]**2)) 
QFea4 = QFea4.T

QFea5 = np.vstack((
np.ones(Fea5[:,0].size), # add the bias term
Fea5[:,0],Fea5[:,1], Fea5[:,0]*Fea5[:,1],
Fea5[:,0]**2,Fea5[:,1]**2)) 
QFea5 = QFea5.T

QFea6 = np.vstack((
np.ones(Fea6[:,0].size), # add the bias term
Fea6[:,0],Fea6[:,1], Fea6[:,0]*Fea6[:,1],
Fea6[:,0]**2,Fea6[:,1]**2)) 
QFea6 = QFea6.T

classfierQ1 = linear_model.LogisticRegression()
classfierQ2 = linear_model.LogisticRegression()
classfierQ3 = linear_model.LogisticRegression()
classfierQ4 = linear_model.LogisticRegression()
classfierQ5 = linear_model.LogisticRegression()
classfierQ6 = linear_model.LogisticRegression()
Qa1=classfierQ1.fit(QFea1,Lab1)
Qa2=classfierQ2.fit(QFea2,Lab2)
Qa3=classfierQ3.fit(QFea3,Lab3)
Qa4=classfierQ4.fit(QFea4,Lab4)
Qa5=classfierQ5.fit(QFea5,Lab5)
Qa6=classfierQ6.fit(QFea6,Lab6)

## x is d*N matrix, Parameters is 1*d matrix, return 1*N matrix with Probabilities for each test points
def Prob(Parameters,x):
    Proba = 1/(1+np.exp(-Parameters*x))
    return Proba
# second and third column are the raw features corresponding to Fea[:,1] and Fea[:,2]
def plotDB_QuadraticFea(Fea=QFea1,Lab=Lab1,LogiObj = Qa1):
# create the domain for the plot
    x_min = min(Fea[:,1])-0.5; x_max = max(Fea[:,1])+0.5
    y_min = min(Fea[:,2])-0.5; y_max = max(Fea[:,2])+0.5

    x1 = np.linspace(x_min, x_max, 200)
    y1 = np.linspace(y_min, y_max , 200)
    x,y = np.meshgrid(x1, y1)
    
# make a 3 x N matrix of the sample points
    data1 = np.vstack((
    np.ones(x.size), # add the bias term
    x.ravel(), # make the matrix into a vector
    y.ravel(), 
    x.ravel()*y.ravel(),
    x.ravel()**2,
    y.ravel()**2)) # add a quadratic term for fun


    Parameters1 = np.matrix(LogiObj.coef_)
    z1 = Prob(Parameters=Parameters1,x=data1)
    z1 = z1.reshape(x.shape)

# Make the plots
# show the function value in the background
    cs = plt.imshow(z1,
        extent=(x_min,x_max,y_max,y_min), # define limits of grid, note reversed y axis
        cmap=plt.cm.jet)
    plt.clim(0,1) # defines the value to assign the min/max color

# draw the line on top
    levels = np.array([.5])
    cs_line = plt.contour(x,y,z1,levels)

# add a color bar
    CB = plt.colorbar(cs)

# add data points
    for lab,x1t,x2t in zip(Lab,Fea[:,1],Fea[:,2]):
        if lab==1:
            plt.scatter(x1t,x2t,marker='v')
        else:
            plt.scatter(x1t,x2t,marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('visualization of classifier, implimentation #2,Quadratic') 
    plt.show()

plotDB_QuadraticFea(Fea=QFea1,Lab=Lab1,LogiObj = Qa1)
Get_err(Features=QFea1,Labels=Lab1,LogiObj=Qa1)

plotDB_QuadraticFea(Fea=QFea2,Lab=Lab2,LogiObj = Qa2)
Get_err(Features=QFea2,Labels=Lab2,LogiObj=Qa2)

plotDB_QuadraticFea(Fea=QFea3,Lab=Lab3,LogiObj = Qa3)
Get_err(Features=QFea3,Labels=Lab3,LogiObj=Qa3)

plotDB_QuadraticFea(Fea=QFea4,Lab=Lab4,LogiObj = Qa4)
Get_err(Features=QFea4,Labels=Lab4,LogiObj=Qa4)

plotDB_QuadraticFea(Fea=QFea5,Lab=Lab5,LogiObj = Qa5)
Get_err(Features=QFea5,Labels=Lab5,LogiObj=Qa5)

plotDB_QuadraticFea(Fea=QFea6,Lab=Lab6,LogiObj = Qa6)  
Get_err(Features=QFea6,Labels=Lab6,LogiObj=Qa6) 


##3 Variable Features Space for doing the logistic regression
##I fix the number of features to be numofFea, FeaAll is all the features I put in the function, we are trying to find the index of all features that minimize 
##the misclassification err.  
def bestDB(numofFea=4,FeaAll = QFea1,Lab=Lab1):
    D = FeaAll.shape[1]
    ind = list(range(D))
    allComb = list(combinations(ind,numofFea))
    index = 0
    err = 1000000000
    for tempInd in allComb:
        TempFea = FeaAll[:,tempInd]
        classfierQ = linear_model.LogisticRegression()
        TempObj = classfierQ.fit(TempFea,Lab)
        tempErr = Get_err(Features=TempFea,Labels=Lab,LogiObj=TempObj)
        if err>tempErr:
            err = tempErr
            index = tempInd
    return err,index        


def plotDB_VariableFea(Fea=QFea6,Lab=Lab6,index = f6[1]):
# create the domain for the plot
    x_min = min(Fea[:,1])-0.5; x_max = max(Fea[:,1])+0.5
    y_min = min(Fea[:,2])-0.5; y_max = max(Fea[:,2])+0.5

    x1 = np.linspace(x_min, x_max, 200)
    y1 = np.linspace(y_min, y_max , 200)
    x,y = np.meshgrid(x1, y1)
    
# make a 3 x N matrix of the sample points
    data1 = np.vstack((
    np.ones(x.size), # add the bias term
    x.ravel(), # make the matrix into a vector
    y.ravel(), 
    x.ravel()*y.ravel(),
    x.ravel()**2,
    y.ravel()**2)) # add a quadratic term for fun
    
    data1 = data1[index,:]
    classifier = linear_model.LogisticRegression()
    Fea0 = Fea[:,index]
    LogiObj=classfier.fit(Fea0,Lab)
    Parameters1 = np.matrix(LogiObj.coef_)
    z1 = Prob(Parameters=Parameters1,x=data1)
    z1 = z1.reshape(x.shape)
# Make the plots
# show the function value in the background
    cs = plt.imshow(z1,
        extent=(x_min,x_max,y_max,y_min), # define limits of grid, note reversed y axis
        cmap=plt.cm.jet)
    plt.clim(0,1) # defines the value to assign the min/max color

# draw the line on top
    levels = np.array([.5])
    cs_line = plt.contour(x,y,z1,levels)

# add a color bar
    CB = plt.colorbar(cs)

# add data points
    for lab,x1t,x2t in zip(Lab,Fea[:,1],Fea[:,2]):
        if lab==1:
            plt.scatter(x1t,x2t,marker='v')
        else:
            plt.scatter(x1t,x2t,marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('visualization of classifier,with the best subset') 
    plt.show()

f1 = bestDB(numofFea=4,FeaAll=QFea1,Lab=Lab1)
plotDB_VariableFea(Fea=QFea1,Lab=Lab1,index = f1[1])

f2 = bestDB(numofFea=4,FeaAll=QFea2,Lab=Lab2)
plotDB_VariableFea(Fea=QFea2,Lab=Lab2,index = f2[1])

f3 = bestDB(numofFea=4,FeaAll=QFea3,Lab=Lab3)
plotDB_VariableFea(Fea=QFea3,Lab=Lab3,index = f3[1])

f4 = bestDB(numofFea=5,FeaAll=QFea4,Lab=Lab4)
plotDB_VariableFea(Fea=QFea4,Lab=Lab4,index = f4[1])

f5 = bestDB(numofFea=4,FeaAll=QFea5,Lab=Lab5)
plotDB_VariableFea(Fea=QFea5,Lab=Lab5,index = f5[1])

f6 = bestDB(numofFea=4,FeaAll=QFea6,Lab=Lab6)
plotDB_VariableFea(Fea=QFea6,Lab=Lab6,index = f6[1])



