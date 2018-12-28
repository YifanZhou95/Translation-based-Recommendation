# -*- coding: utf-8 -*-
"""
Factorizing Personalized Markov Chain

Created on Tue Jan 16 14:42:53 2018

@author: yfzhou
"""

import numpy as np
import random
from math import exp
from math import log
import matplotlib.pyplot as plt

dataset_name = 'Video_Games'
dataset = np.load('../data/'+dataset_name+'Partitioned.npy')
[user_train,user_validation,user_test, usernum,itemnum] = dataset

item_successor = [[] for it in range(itemnum)]
for user in user_train:
    for i in range(len(user_train[user])-1):
        pre = user_train[user][i]
        suc = user_train[user][i+1]
        item_successor[pre].append(suc)

num_relation = sum([len(item_successor[item]) for item in range(itemnum)])

def sigmoid(x):
    return 1.0 / (1 + exp(-x))

def findUser():
    while 1:
        user = random.randint(0,usernum-1)
        if len(user_train[user]) > 1:
            return user

def findNegSucc(pos_item):
    while 1:
        neg_item = random.randint(0,itemnum-1)
        if neg_item != pos_item:
            return neg_item

def FPMCPredict(user, pre, cur):
    return np.dot(gamU[user,:],gamI[:,cur]) + np.dot(kap[pre,:],eta[:,cur])

def AUC():
    auc_train = 0
    auc_valid = 0
    auc_test = 0
    testnum = 0     # event num per user in AUC testing
    # max_itemid = max(item_train.keys())
    for user in user_test:
        if len(user_train[user])<2 or len(user_test[user])==0:
            continue
        testnum += 1

        train_pre_item = user_train[user][-2]
        train_item = user_train[user][-1]
        train_score = FPMCPredict(user, train_pre_item, train_item)          
        
        valid_pre_item = user_validation[user][0]
        valid_item = user_validation[user][1]
        valid_score = FPMCPredict(user, valid_pre_item, valid_item)
        
        test_pre_item = user_test[user][0]
        test_item = user_test[user][1]
        test_score = FPMCPredict(user, test_pre_item, test_item)
        
        
        count_train, count_valid, count_test = 0, 0, 0
        neg_num = 0
        for ind in range(100):
            itemid = random.randint(0,itemnum-1)
            if itemid not in user_train[user] and itemid not in user_test[user]:
                neg_num += 1
                
                neg_score = FPMCPredict(user, train_pre_item, itemid)
                if neg_score < train_score:
                    count_train += 1
                elif neg_score == valid_score:
                    count_train += 0.5
                else:
                    count_train += 0                
                
                neg_score = FPMCPredict(user, valid_pre_item, itemid)
                if neg_score < valid_score:
                    count_valid += 1
                elif neg_score == valid_score:
                    count_valid += 0.5
                else:
                    count_valid += 0
                
                neg_score = FPMCPredict(user, test_pre_item, itemid)
                if neg_score < test_score:
                    count_test += 1
                elif neg_score == test_score:
                    count_test += 0.5
                else:
                    count_test += 0
        
        auc_train += count_train*1.0 / neg_num
        auc_valid += count_valid*1.0 / neg_num
        auc_test += count_test*1.0 / neg_num
    
    auc_train = auc_train/testnum
    auc_valid = auc_valid/testnum
    auc_test = auc_test/testnum
    print ("training AUC: ", auc_train)
    print ("validation AUC: ", auc_valid)
    print ("testing AUC: ", auc_test)
    return auc_train, auc_valid, auc_test

lam = 0.01
K1 = 10
K2 = 10
learn_rate = 0.05
max_iter = 1000
gamU = np.random.rand(usernum, K1)/1 - 0.5
gamI = np.random.rand(K1, itemnum)/1 - 0.5
kap = np.random.rand(itemnum, K2)/1 - 0.5
eta = np.random.rand(K2, itemnum)/1 - 0.5

auc_rec_train = []
auc_rec_valid = []
auc_rec_test = []

for it in range(max_iter):
    objective = 0
    regularization = 0
#    dg = np.zeros((itemnum, K))
#    de = np.zeros((K, itemnum))
    
    for ind in range(num_relation):
        u = findUser()
        position = random.randint(0,len(user_train[u])-2)
        p = user_train[u][position]        # previous item
        i = user_train[u][position + 1]    # positive item
        j = findNegSucc(i)                 # negative item
        z = sigmoid(np.dot(gamU[u,:],gamI[:,i]) - \
                    np.dot(gamU[u,:],gamI[:,j]) + \
                    np.dot(kap[p,:],eta[:,i]) - \
                    np.dot(kap[p,:],eta[:,j]))
#        dg[u,:] += (1-z)*(eta[:,i]-eta[:,j])
#        de[:,i] += (1-z)*(gam[u,:])
#        de[:,j] += (1-z)*(-gam[u,:])
        gamU[u,:] += learn_rate*((1-z)*(gamI[:,i]-gamI[:,j]) - 2*lam*gamU[u,:])
        gamI[:,i] += learn_rate*((1-z)*(gamU[u,:]) - 2*lam*gamI[:,i])
        gamI[:,j] += learn_rate*((1-z)*(-gamU[u,:]) - 2*lam*gamI[:,j])
        kap[p,:] =+ learn_rate*((1-z)*(eta[:,i]-eta[:,j])-2*lam*kap[p,:])
        eta[:,i] += learn_rate*((1-z)*(kap[p,:]) - 2*lam*eta[:,i])
        eta[:,j] += learn_rate*((1-z)*(-kap[p,:]) - 2*lam*eta[:,j])
        
        objective += log(z)
    
#    dg -= lam*gam
#    de -= lam*eta
#    gam += learn_rate*dg
#    eta += learn_rate*de 
    
    regularization = objective - lam*np.sum(np.square(gamU)) - \
                                 lam*np.sum(np.square(gamI)) - \
                                 lam*np.sum(np.square(kap)) - \
                                 lam*np.sum(np.square(eta))
                                                 
    if (it+1)%5 == 0:
        print ('iteration: ' + str(it+1) + '\t' + str(regularization) \
                                         + '\t' + str(objective))
    if (it+1)%10 == 0:
        auc = AUC()
        auc_rec_train.append(auc[0])
        auc_rec_valid.append(auc[1])
        auc_rec_test.append(auc[2])

plt.figure()
plt.plot(auc_rec_train)
plt.figure()
plt.plot(auc_rec_valid)
plt.figure()
plt.plot(auc_rec_test)


