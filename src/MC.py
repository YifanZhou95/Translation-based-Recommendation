# -*- coding: utf-8 -*-
"""
Markov Chain Recommendation

Created on Sun Jan 14 11:33:09 2018

@author: yfzhou
"""

import numpy as np
import random
from math import exp
from math import log


dataset_name = 'Automotive'
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

def findItem():
    while 1:
        item = random.randint(0,itemnum-1)
        if item_successor[item] != []:
            return item

def findNegSucc(item):
    while 1:
        neg_item = random.randint(0,itemnum-1)
        if neg_item not in item_successor[item]:
            return neg_item

def MCPredict(user, pre, cur):
    return np.dot(gam[pre,:],eta[:,cur])

def AUC():
    auc_valid = 0
    auc_test = 0
    testnum = 0
    # max_itemid = max(item_train.keys())
    for user in user_test:
        if len(user_test[user])==0:
            continue
        testnum += 1
        
        valid_pre_item = user_validation[user][0]
        valid_item = user_validation[user][1]
        valid_score = MCPredict(user, valid_pre_item, valid_item)        
        test_pre_item = user_test[user][0]
        test_item = user_test[user][1]
        test_score = MCPredict(user, test_pre_item, test_item)
        count_valid = 0
        count_test = 0
        neg_num = 0
        for ind in range(100):
            itemid = random.randint(0,itemnum-1)
            if itemid not in user_train[user] and itemid not in user_test[user]:
                neg_num += 1
                neg_score = MCPredict(user, valid_pre_item, itemid)
                if neg_score < valid_score:
                    count_valid += 1
                elif neg_score == valid_score:
                    count_valid += 0.5
                else:
                    count_valid += 0
                
                neg_score = MCPredict(user, test_pre_item, itemid)
                if neg_score < test_score:
                    count_test += 1
                elif neg_score == test_score:
                    count_test += 0.5
                else:
                    count_test += 0
        
        auc_valid += count_valid*1.0 / neg_num
        auc_test += count_test*1.0 / neg_num
    
    auc_valid = auc_valid/testnum
    auc_test = auc_test/testnum
    print "validation AUC: ", auc_valid
    print "testing AUC: ", auc_test
    # return auc_valid

lam = 0.1
K = 10
learn_rate = 0.2
max_iter = 200

gam = np.random.rand(itemnum, K)/1 - 0.5
eta = np.random.rand(K, itemnum)/1 - 0.5
accRec = []

for it in range(max_iter):
    objective = 0
    regularization = 0
#    dg = np.zeros((itemnum, K))
#    de = np.zeros((K, itemnum))
    
    for ind in range(num_relation):
        u = findItem()
        i = random.choice(item_successor[u])
        j = findNegSucc(u)
        z = sigmoid(np.dot(gam[u,:],eta[:,i]) - \
                    np.dot(gam[u,:],eta[:,j]))
#        dg[u,:] += (1-z)*(eta[:,i]-eta[:,j])
#        de[:,i] += (1-z)*(gam[u,:])
#        de[:,j] += (1-z)*(-gam[u,:])
        gam[u,:] += learn_rate*((1-z)*(eta[:,i]-eta[:,j])-2*lam*gam[u,:])
        eta[:,i] += learn_rate*((1-z)*(gam[u,:]) - 2*lam*eta[:,i])
        eta[:,j] += learn_rate*((1-z)*(-gam[u,:]) - 2*lam*eta[:,j])
        
        objective += log(z)
    
#    dg -= lam*gam
#    de -= lam*eta
#    gam += learn_rate*dg
#    eta += learn_rate*de 
    
    regularization = objective - lam*np.sum(np.square(gam)) - \
                                       lam*np.sum(np.square(eta))        
    if (it+1)%2 == 0:
        print 'iteration: ' + str(it+1) + '\t' + str(regularization) \
                                 + '\t' + str(objective)
    if (it+1)%10 == 0: AUC()

