# -*- coding: utf-8 -*-
"""
Popularity recommendation

Created on Tue Jan 09 09:33:06 2018

@author: zyf
"""

import numpy as np
import random
from collections import defaultdict
from sklearn import metrics

dataset_name = 'Office'
dataset = np.load('../data/'+dataset_name+'Partitioned.npy')
[user_train,user_validation,user_test, usernum,itemnum] = dataset

# merge training set and validation set in PopRec setting
#for user in user_train:
#    if user_validation[user] != []:
#        user_train[user].append(user_validation[user][1])

def PopModel(user_train):
    item_train = {}
    for user in user_train:
        for itemid in user_train[user]:
            if itemid in item_train:
                item_train[itemid] += 1
            else:
                item_train[itemid] = 1
    return item_train
# most_pop = sorted(item_train,key=lambda x:item_train[x])[-1]

def PopPredict(item_train, query_item):
    return item_train[query_item] if query_item in item_train else 0

def AUC(dataset, item_train):
    [user_train,user_validation,user_test, usernum,itemnum] = dataset
    auc_total = 0
    testnum = 0
    # max_itemid = max(item_train.keys())
    for user in user_test:
        if len(user_test[user])==0:
            continue
        testnum += 1
        
        test_item = user_test[user][1]
        test_score = PopPredict(item_train, test_item)
        count = 0
        neg_num = 0
        for ind in range(500):
            itemid = random.randint(0,itemnum)
            if itemid not in user_train[user] and itemid not in user_test[user]:
                neg_num += 1
                neg_score = PopPredict(item_train, itemid)
                if neg_score < test_score:
                    count += 1
                elif neg_score == test_score:
                    count += 0.5
                else:
                    count += 0
    
        auc_total += count*1.0/neg_num
        
    #    neg_dict = {x:item_train[x] for x in item_train}
    #    for itemid in user_train[user]:
    #        if itemid in neg_dict:
    #            del neg_dict[itemid]
    #    for itemid in user_test[user]:
    #        if itemid in neg_dict:
    #            del neg_dict[itemid]
    #
    #    neg_list = neg_dict.values()
    #    lable_list = [0 for x in neg_list] + [1]
    #    test_item = user_test[user][1]
    #    test_score = item_train[test_item] if test_item in item_train else 0
    #    score_list = neg_list + [test_score]
    #    fpr, tpr, thresholds = metrics.roc_curve(lable_list, score_list, pos_label=1)
    #    auc_total += metrics.auc(fpr, tpr)
        
        if user%1000 == 0:
            print user/1000
    
    auc_ave = auc_total/testnum
    print "AUC: ", auc_ave

    
item_train = PopModel(user_train)
AUC(dataset, item_train)
