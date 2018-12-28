
# coding: utf-8

# @author: wang-cheng kang

# In[1]:

import gzip
from collections import defaultdict
from datetime import datetime


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


countU = defaultdict(lambda: 0)
countP = defaultdict(lambda: 0)
line=0

dataset_name = 'Cell_Phones'
f=open('reviews_'+dataset_name+'.txt','w')
for l in parse('reviews_'+dataset_name+'.json.gz'):
    line+=1
    f.write(" ".join([l['reviewerID'],l['asin'],str(l['overall']),str(l['unixReviewTime'])])+' \n')
    asin = l['asin']    # gPlusPlaceId
    rev = l['reviewerID'] 
    time = l['unixReviewTime']
    countU[rev]+=1
    countP[asin]+=1    
f.close()

# del all_asin
# del imgavailable_asin
# del cnnavailable_asin
    
 


# In[2]:

usermap = dict()
usernum = 0
itemmap = dict()
itemnum = 0
User = dict()
for l in parse('reviews_'+dataset_name+'.json.gz'):
    line+=1
    asin = l['asin']
    rev = l['reviewerID'] 
    time = l['unixReviewTime']
    if countU[rev]<5 or countP[asin]<5 : continue
        
    if rev in usermap: userid = usermap[rev]
    else:
        userid = usernum
        usernum += 1
        usermap[rev] = userid
        User[userid] = []
    if asin in itemmap: itemid = itemmap[asin]
    else:
        itemid = itemnum
        itemnum += 1
        itemmap[asin] = itemid    
    User[userid].append([time,itemid])
# sort reviews in User according to time

for userid in User.keys():
#     #User[userid].sort()
    User[userid].sort(key=lambda x: x[0])


# In[3]:

User[0]


# In[4]:

IIG = dict()
Relationships = ['also_bought','also_viewed','bought_together','buy_after_viewing']
for l in parse('meta_'+dataset_name+'.json.gz'):
   asin = l['asin']
   if not asin in itemmap: continue
   itemid = itemmap[asin]
   IIG[itemid] = dict()
   if 'related' in l:
       IIG[itemid]['related'] = dict()
       for rel in Relationships:
           IIG[itemid]['related'][rel] = []
           if rel in l['related']:
               for n_asin in l['related'][rel]:
                   if n_asin in itemmap: IIG[itemid]['related'][rel].append(itemmap[n_asin])
               
   else:
       IIG[itemid]['related'] = dict()
       for rel in Relationships: IIG[itemid]['related'][rel] = []
   IIG[itemid]['categories'] = l['categories']


# In[5]:

import numpy as np
dataset=[User,IIG,usermap,itemmap,usernum,itemnum]
np.save('meta_'+dataset_name+'.npy',dataset)


# In[9]:


