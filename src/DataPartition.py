
# coding: utf-8

# @author: wang-cheng kang, yfzhou

# In[24]:

import gzip
import math
import json
import ujson
import random
import numpy as np

dataset_name = 'Video_Games'
dataset=np.load('meta_'+dataset_name+'.npy')

[User,Item,usermap,itemmap,usernum,itemnum]=dataset


# In[25]:

for user in User:
    User[user] = [b for a,b in User[user]]
print usernum,itemnum
t=0
for user in User: t+=len(User[user])
print t 


# In[26]:

#t=0
#for item in Item:
#    for k in Item[item]['related']:
#        t+=len(Item[item]['related'][k])
#t    


# In[3]:

#devided to train/validation/test 
user_train=dict()
user_validation=dict()
user_test=dict()

for user in User:
    nfeedback=len(User[user])
    if nfeedback<3:
        user_train[user]=User[user]
        user_validation[user]=[]
        user_test[user]=[]
    else:
        user_train[user]=User[user][:-2]
        user_validation[user]=[]
        user_validation[user].append(User[user][-3])
        user_validation[user].append(User[user][-2])
        user_test[user]=[]
        user_test[user].append(User[user][-2])
        user_test[user].append(User[user][-1])

        
    


# In[4]:

user_train[2]


# In[5]:
import numpy as np
dataset=[user_train,user_validation,user_test, usernum,itemnum]  # exclude item
np.save('../data/'+dataset_name+'Partitioned.npy',dataset)


# In[9]:

for line in file('samples.txt'):
    u,i,j,jp = map(int,line.rstrip().split(','))
    f=False
    s=set()
    for I in range(len(user_train[u])-1):
        s.add(user_train[u][I])
        s.add(user_train[u][I+1])
        if user_train[u][I]==i and user_train[u][I+1]==j: f=True
    if (not f) or (jp in s):
        print u,i,j,jp
        print user_train[u]


# In[8]:

# dataset=[user_train,user_validation,user_test,Item,usernum,itemnum]
# np.save('../AmazonFashionpricePartioned.npy',dataset)

