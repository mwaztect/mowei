# Author: Mo Wei
# Email : wei.mo@digpenedu
# Descriptopn : Python Code with K-Means Algorhtim implemented
#               As Per AM Practicum 01 Requirement  

import numpy as np
import re
import math
import random
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns


def loadData(filename):
    #load dataset
    X=[]
    count = 0

    text_file = open(filename, "r")
    lines = text_file.readlines()
    
    for line in lines:
        X.append([])
        words = re.split('[ \t]+',line.strip())
        #print words  
        for word in words:
            X[count].append(float(word))
        count += 1
    return np.asarray(X)


def errCompute(X, M):
    '''
    calculate the value of objective function for the clustering
    Input: dataset and cluster ID, mean value for each cluster
    Output: value of objective function
    '''
    err =0
    for idx, x in enumerate(X):
        err+= math.sqrt((x[0]-M[int(x[2]),0])**2 + (x[1]-M[int(x[2]),1])**2) 
    return err/X.shape[0]

def Group(X, M):
    '''
    dataset X with updated cluster ID, assign each object into its closest cluster
    Input: dataset and cluster ID, current mean value for each cluster
    Output: Assign X with updated cluster ID 
    '''
    # for i in range(X.shape[0]):
    #     dis = math.sqrt((X[i,0]-M[int(X[i,2]),0])**2+(X[i,1]-M[int(X[i,2]),1])**2)
    #     for k in range(M.shape[0]):
    #         if math.sqrt((X[i,0]-M[k,0])**2+(X[i,1]-M[k,1])**2)<dis:
    #             X[i,2] = k
    #             dis = math.sqrt((X[i,0]-M[k,0])**2+(X[i,1]-M[k,1])**2)
    # return X
    X= X.copy()
    M = M.copy()
    coords = np.expand_dims(X[:,:2],axis=1)
    distance_vect = coords-M
    distance = np.linalg.norm(distance_vect,axis=2)
    X[:,2] = np.argmin(distance,axis=1)
    return X


def calcMeans(X, M):
    '''
    update mean value for each cluster
    Input: dataset and cluster ID, current mean value for each cluster
    Output: updated mean value
    '''
# #     cluster = np.unique(X[:,2])
#     sumval = np.zeros(M.shape)
#     cnt = np.zeros(M.shape[0])
#     for i in range(X.shape[0]):
#         for k in range(M.shape[0]):
#             if round(X[i,2],2)==float(k):
#                 sumval[k,0]+=X[i,0]
#                 sumval[k,1]+=X[i,1]
#                 cnt[k]+=1
    
#     for k in range(M.shape[0]):
#         M[k,0]=sumval[k,0]/cnt[k]
#         M[k,1]=sumval[k,1]/cnt[k]
    X=np.copy(X)
    M=np.copy(M)
    for cluster in range(M.shape[0]):
        new = X[X[:,2]==float(cluster)]
        M[cluster,0:2]=np.mean(new[:,0:2],axis=0)

    return M

def plotgraph(X, K, task):
    '''Plot graph for latitude vs. longtitude'''
    plt.figure(figsize=(12,8))
    for i in range(K):
        newArr = X[X[:,2] == float(i)]
        plt.scatter(newArr[:,0],newArr[:,1], label=f'Cluster {i}')   
    plt.title(f'Plot of dataset with K = {K}')
    plt.xlabel('Latitude')
    plt.ylabel('longitude')
#     plt.grid()
    # plt.show()
    plt.savefig(f'output/task_{task}.png')
    print(f'\nFigure saved into output folder as: output/task_{task}.png\n')
    return

def kmeans(X,K,M,task):
    '''
    Task g h i: implementation of K-Means algorithm
    Input: dataset X, number of cluster K, initial cluster mean and task name
    Output: figure plot for each task'''

    iteration = 0
#     start = time.time()
    
    while True:
        iteration+=1
        clus_pre = np.copy(X[:,2])
        X = Group(X, M)
        if np.all(np.equal(clus_pre, X[:,2])):
#             print(f'Break at iteration: {iteration}')
            break
        M = calcMeans(X, M)
#     stop = time.time()
    err = errCompute(X, M)
    print(f'Error of task {task} is: ', err)
#     print(f'Computing time: ', round((stop-start),5))
    plotgraph(X,K,task)
    return err


if __name__ == '__main__':

    print('\n', '*'*20, 'Task a', '*'*20, '\n')
    filename='2010825.txt'
    K=5
    X = loadData(filename)
    X = np.c_[X,np.random.randint(1, size=X.shape[0])]
    # print(X.shape)
    np.savetxt(f'iniMeans/Initial_X.txt',np.asarray(X), fmt='%.8f', delimiter='  ', newline='\n')
    print(f'\nX shape after storing cluster ID: {X.shape}.\nInitial dataset saved in "iniMeans" folder as: "Initial_X.txt"\n')
    M=np.copy(X[0:K,0:X.shape[1]-1])
    np.savetxt(f'iniMeans/Initial_M_k=5.txt',np.asarray(M), fmt='%.8f', delimiter='  ', newline='\n')
    print(f'\nCluster M shape after storing cluster ID: {M.shape}.\nInitial cluster mean saved in "iniMeans" folder as: "Initial_M_k=5.txt"\n')

    print('\n', '*'*20, 'Task b', '*'*20, '\n')
    plotgraph(X,1,'b')

    print('\n', '*'*10, 'Task c as stated in report section 2', '*'*10, '\n')

    print('\n', '*'*20, 'Task d', '*'*20, '\n')
    Md = np.array([[0,0]])
    errd = errCompute(X, Md)
    print(f'\nError of task d is: {errd}.\n')

    print('\n', '*'*10, 'Task e as stated in report section 3', '*'*10, '\n')

    print('\n', '*'*20, 'Task f', '*'*20, '\n')
    X = Group(X, M)
    errf = errCompute(X, M)
    print(f'\nError of task f is: {errf}.\n')

    print('\n', '*'*10, 'Task g as stated in report section 4', '*'*10, '\n')
    
    print('\n', '*'*20, 'Task h', '*'*20, '\n')
    print('\nCalculating distance error with K=5...\n')
    kmeans(X,K,M,'h')

    print('\n', '*'*20, 'Task i K=50', '*'*20, '\n')
    ki1=50
    Mi1=np.copy(X[0:ki1,0:X.shape[1]-1])
    np.savetxt(f'iniMeans/Initial_M_k=50.txt',np.asarray(Mi1), fmt='%.8f', delimiter='  ', newline='\n')
    print(f'\nCluster mean for K=50 saved in "iniMeans" folder as: "Initial_M_k=50.txt"\n')
    print('\nCalculating distance error with K=50...\n')
    kmeans(X,ki1,Mi1,'i_k=50')

    print('\n', '*'*20, 'Task i K=100', '*'*20, '\n')
    ki2=100
    Mi2=np.copy(X[0:ki2,0:X.shape[1]-1])
    np.savetxt(f'iniMeans/Initial_M_k=100.txt',np.asarray(Mi1), fmt='%.8f', delimiter='  ', newline='\n')
    print(f'\nCluster mean for K=100 saved in "iniMeans" folder as: "Initial_M_k=100.txt"\n')
    print('\nCalculating distance error with K=100...\n')
    kmeans(X,ki2,Mi2,'i_k=100')










