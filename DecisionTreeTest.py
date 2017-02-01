# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 14:37:46 2015

@author: SSL
"""
#numofCut is the number of cutoff we choose for each step,data should be n*3 dataframe or ndarray
#ind need to be 0 or 1 for feature1 or feature2
def onesplit(TPoints,numofCut,ind=1):
    if isinstance(TPoints,pd.DataFrame):
        TPoints = TPoints.as_matrix()
    GiniInd = []
    n=len(TPoints)
    cut1 = np.linspace(min(TPoints[:,ind]),max(TPoints[:,ind]),num=numofCut)
    for c1 in cut1:               
        a1 = (TPoints[:,ind]>=c1)                    
        a2 = (TPoints[:,ind]<c1)
        node1 = TPoints[a1]
        node2 = TPoints[a2]
        if node1.size==0 or node2.size==0:
            continue
        G1 = 1 - sum((pd.value_counts(node1[:,2])/len(node1))**2) 
        G2 = 1 - sum((pd.value_counts(node2[:,2])/len(node2))**2) 
        Gini = (G1*len(node1) + G2*len(node2))/n
        TempGiniInd = np.array([Gini,c1])
        GiniInd = np.append(GiniInd,TempGiniInd)
    GiniInd = GiniInd.reshape((len(GiniInd)/2,2))                    
    OptGini = GiniInd[np.where(GiniInd[:,0]==min(GiniInd[:,0]))]      # minimize the Gini impurity
    Optcut = float(OptGini[:,1][0])         ##OptGini may have several choice, I take the first one
    TreeLft = TPoints[TPoints[:,ind] < Optcut]
    TreeRig = TPoints[TPoints[:,ind] >= Optcut]
    return(Optcut,TreeLft,TreeRig)

#generating Decision tree with depth 3
#testPoints is a n*2 ndarray for prediction
def DTtree(TrainP,numofCut,testPoints):
    X1results = onesplit(TPoints=TrainP,numofCut=numofCut,ind=0)
    OptCut1 = X1results[0]
    DataL = X1results[1]
    DataR = X1results[2]
    X2resultL = onesplit(TPoints=DataL,numofCut=numofCut,ind=1)
    X2resultR = onesplit(TPoints=DataR,numofCut=numofCut,ind=1)
    OptCut2L = X2resultL[0]
    OptCut2R = X2resultR[0]
    Node_LL = X2resultL[1]
    Node_LR = X2resultL[2]
    Node_RL = X2resultR[1]
    Node_RR = X2resultR[2]
    Label_LL = 0 if sum(Node_LL[:,2])/len(Node_LL[:,2]) <0.5 else 1
    Label_LR = 0 if sum(Node_LR[:,2])/len(Node_LR[:,2]) <0.5 else 1
    Label_RL = 0 if sum(Node_RL[:,2])/len(Node_RL[:,2]) <0.5 else 1
    Label_RR = 0 if sum(Node_RR[:,2])/len(Node_RR[:,2]) <0.5 else 1
    predi = []
    for TestP in testPoints:
        TestP = list(TestP)
        x1 = TestP[0]
        x2 = TestP[1]
        if x1<OptCut1 and x2<OptCut2L:
            predi.append(Label_LL)
        elif x1<OptCut1 and x2>=OptCut2L:
            predi.append(Label_LR)
        elif x1>=OptCut1 and x2<OptCut2R:
            predi.append(Label_RL)
        else:
            predi.append(Label_RR)  
    predi = np.asarray(predi)
    return{'cut':[OptCut1,OptCut2L,OptCut2R],'prediction':predi}
         
            
#use training data to get the training set error
pred = DTtree(TrainP=syn1,numofCut=40,testPoints=syn1.values[:,0:2])['prediction']
TrainErr = 1- sum(pred==syn1.values[:,2])/len(syn1)            
            
pred = DTtree(TrainP=syn2,numofCut=40,testPoints=syn2.values[:,0:2])['prediction']
TrainErr = 1- sum(pred==syn2.values[:,2])/len(syn2)   

pred = DTtree(TrainP=syn3,numofCut=40,testPoints=syn3.values[:,0:2])['prediction']
TrainErr = 1- sum(pred==syn3.values[:,2])/len(syn3)   

pred = DTtree(TrainP=syn4,numofCut=40,testPoints=syn4.values[:,0:2])['prediction']
TrainErr = 1- sum(pred==syn4.values[:,2])/len(syn4)   
            
            
   