import csv
import json
import pandas as pd
from collections import defaultdict
import os
import itertools
import random

#splits list into pairs of two
def gimme_pairs(li):
	#if not even sized then replicate last utt and add as last slot:
	if (len(li) % 2 !=0):
		li=li+ [ li[-1] ]

	for i in range(0, len(li), 2):
		 yield li[i:i+2]


train='8_slot_mr_test.csv'
test='data/devset.csv'
print("Opening file "+train)
df_train=pd.read_csv(train)
df_test=pd.read_csv(test)


json_train=dict()
#train conversion
for index, value in df_train.iterrows():
	json_train.setdefault(value['mr'], []).append(value['ref'])

#print json.dumps(json_train, indent=4, ensure_ascii=False)
print("Converting train file: "+ train)
train_list=[] #total list
for k,v in json_train.iteritems():
	k="".join(i for i in k if ord(i)<128) #remove non ascii
	k=k.replace("[","='")
	k=k.replace("]", "'")
	k=k.replace(",", ";")
	k= "inform("+k+")"
	pairs = gimme_pairs(v)
	for utt in pairs: 
		train_list= train_list+[ [k]+utt ]

#30% validation, rest train
#print("permuting data. creating train and validation sets...")
#random.shuffle(train_list)
#n_points=len(train_list)
#break_point=int(0.3*n_points)
#valid_set=train_list[: break_point]
#train_set=train_list[break_point: ]


print("Writting file to .json...")
#print json.dumps(train_list, indent=4, ensure_ascii=False)
with open('JsonData/train8/test.json', 'w') as f:
	json.dump(train_list, f, indent=4, ensure_ascii=False)

#with open('JsonData/train8/valid.json', 'w') as f:
#	json.dump(valid_set, f, indent=4, ensure_ascii=False)


'''

#Now convert the test
json_test=dict()
for index, value in df_test.iterrows():
	json_test.setdefault(value['mr'], []).append(value['ref'])
	

test_list=[]
for k,v in json_test.iteritems():
	k="".join(i for i in k if ord(i)<128) #remove non ascii
	k=k.replace("[","='")
	k=k.replace("]", "'")
	k=k.replace(",", ";")
	k= "inform("+k+")"
	pairs = gimme_pairs(v)
	for utt in pairs: 
		test_list= test_list+[ [k]+utt ]

with open('data/devset.json', 'w') as f2:
	json.dump(test_list, f2, indent=4, ensure_ascii=False)

#print test_list

'''
