import pandas as pd
import os 
import itertools
from random import shuffle

#set this value to the number of slots you want the MRs to have
##############
n_slots=8
##############

#set this value to True if you want to have ALL permutations of MRs with #slots=n_slots
#CAUTION:IMPACTS PERFORMANCE A LOT (Panos's machine takes about 19sec to run for n_slots=3)
##############
permute=False 
##############


#filter function
#it basically gives the count of all the slots from a given MR
def slot_count(s):
	return len(s.split(",")) 

df=pd.read_csv('data/devset.csv')
#df=pd.read_csv('data/trainset.csv', encoding="latin-1")
df['mr']=df['mr'].astype('str')
df['ref']=df['ref'].astype('str')

n_df=df[df['mr'].map(slot_count)==n_slots]
n_df.to_csv('%d_slot_mr_test.csv' % n_slots, index=False)

num_rows=len(n_df.index)
print ("Number of rows remaining:" + str(num_rows))
count =0
print("program exited before final termination")

if permute:
	perm_df=pd.DataFrame()
	for d_index, row in n_df.iterrows():
		r=row['mr'].split(",")

		if (count == 250):
			print ("Number of rows remaining:" + str(num_rows))
			count=0 #reset count

		num_rows=num_rows-1
		count+=1

		for j in range(0,100):
			shuffle(r)
			df2=pd.DataFrame({'mr': [', '.join(r)],
				'ref':[row['ref']]}
				)
			perm_df=pd.concat([perm_df,df2])

	perm_df.to_csv('perm_%d_slot_mr.csv' % n_slots, index=False)