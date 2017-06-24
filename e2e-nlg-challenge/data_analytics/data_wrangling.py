import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 


pd.options.display.max_colwidth=1000 #for printing

# aux function for filtering strings
def filter_str(s):
	include=True
	new_str=''
	for e in s:

		if e=='[':
			include=False 
		elif e==']':
			include=True

		if include and e!=']':
			new_str+=e
	return new_str



df=pd.read_csv('data/trainset.csv')
mr=df.mr.astype('str') #store column
ref=df.ref.astype('str')

utt_len=ref.str.len() #len of utterances


if os.path.isfile('utt_len.pdf')==False: #do it only if file does not exist
	plt.subplot(1, 2, 1)
	plt.xlim([0,300])
	utt_len.plot.hist(alpha=0.9, range=(0,250), bins=25)
	plt.xlabel("Length of utterances")

	plt.subplot(1, 2, 2)
	color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
	utt_len.plot.box(color=color, sym='r+')
	plt.ylabel("Length")
	plt.xlabel("Panos")

	plt.savefig('utt_len.pdf', bbox_inches='tight')

plt.figure()
plt.xlim([0,300])
plt.xticks(np.arange(0, 300, 20))
utt_len.plot.hist(alpha=0.9, range=(0,250), bins=25)
plt.xlabel("Length of utterances",fontsize=16)
plt.ylabel("Frequency",fontsize=16)
plt.savefig('utt_len.pdf', bbox_inches='tight')

# GROUP BY MR then take count. 
#Note this is the "naive" method since same slots diff values result in different counts
grouped_mr = df.groupby('mr')
get_count=grouped_mr['ref'].agg({ 'DIRECT COUNT': 'count'})

if os.path.isfile('mr_NaiveMultiplicity.pdf')==False: #do it only if file does not exist
	plt.figure()
	h=get_count.plot.hist(alpha=0.9, range=(0,20), bins=20)
	plt.xlabel("# of utterances")
	plt.ylabel("# of MRs")
	plt.savefig('mr_NaiveMultiplicity.pdf', bbox_inches='tight')


#get new df 
df['mr']=df['mr'].apply(lambda s: filter_str(s))
grouped_mr = df.groupby('mr')
get_count=grouped_mr['ref'].agg({ 'ANALOGICAL COUNT': 'count'})


#Example :uncomment to see the one with most utterances
#print get_count[(get_count>80).any(axis=1)] 


if os.path.isfile('mr_Multiplicity.pdf')==False: #do it only if file does not exist
	get_count.to_csv("number_of_occurances.csv", sep=',')



