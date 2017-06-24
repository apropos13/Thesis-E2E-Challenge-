import pandas, csv
import re
import collections
import nltk
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
#from nltk.align.bleu import BLEU

def readData(filename):
	
	df = pandas.read_csv(filename, "IS230", names = ["mr"])
	df_new = pandas.DataFrame(columns=["mr", "ref"])
	#df_new = pandas.DataFrame(columns=["mr", "ref", "generated_utterance"])

	mr = []
	ref = []
	#generated_utterance = []
	for index, row in df.iterrows():
		if index == 0: continue
		
		new_row = row["mr"].split("\",")
		if(len(new_row) == 2):
		#if(len(new_row) == 3):
			mr.append(str(new_row[0]))
			ref.append(str(new_row[1]))
			#generated_utterance.append(str(new_row[2]))
	df_new['mr'] = mr
	df_new['ref'] = ref
	#df_new['generated_utterance'] = generated_utterance
	return(df_new)
	

def generateDict(df):
	mr_dict = collections.defaultdict(dict)

	for index, row in df.iterrows():

		mr = row['mr']
		utterance = row['ref']

		slot_str, attrib_value_dict = getSlotValuePairs(mr)
		mr_dict.update({slot_str:(utterance, attrib_value_dict)})

	return mr_dict


def getSlotValuePairs(mr):
	#test_mr_dict = collections.defaultdict(dict)

	mr = mr.replace("\"", "")
	slots = mr.split(",")

	slot_values= [re.search(r'\[(.*)\]', slot).group(1) for slot in slots]
	slot_attrib = [re.search(r'(.*)\[.*\]', slot).group(1) for slot in slots]

	#print(slot_values)
	#print(slot_attrib)

	attrib_value_dict = {}
	slot_str = ""
		
	for i in range(0, len(slots)):
		attrib_value_dict.update({slot_attrib[i]:slot_values[i]})
		slot_str = slot_str+str(slot_attrib[i])+","
	slot_str = slot_str.strip(",")
	#test_mr_dict.update({slot_str:attrib_value_dict})
	return (slot_str, attrib_value_dict)


def retrieveResponse(mr_dict, test_slot_attrib_value):

	test_slot_str, test_attrib_value_dict = test_slot_attrib_value

	test_slot_str = test_slot_str.split(",")
	test_slot_set = set(test_slot_str)

	for key in mr_dict:
		slot_str = key.split(",")
		slot_set = set(slot_str)
		if(test_slot_set == slot_set):
			utterance, train_dict = mr_dict[key]
			return(modifyUtterance(utterance, train_dict, test_attrib_value_dict))
		continue
	return None
	

def modifyUtterance(utterance, train_dict, test_dict):
	
	for key in train_dict:
		if(train_dict[key] in utterance):
			#print("word in train: "+str(train_dict[key]))
			#print("utterance: "+str(utterance))
			new_value = test_dict[key]
			utterance = utterance.replace(train_dict[key], new_value)
			#print("updated utterance: "+str(utterance))
		
	return utterance


def calculateSlotMatchingRate(reference_utterances, predicted_utterances):
	#The slot matching rate is the percentage of delexicalised tokens (e.g. [s.food] and [v.area] appear in the candidate also appear in the reference.
	#cosine distance between word2vec embeddings
	#generated_dict = find_slot_value_tokens(predicted_utterances)
	pass

if __name__ == "__main__":

	
	train_data = "data_analytics/data/trainset.csv"
	dev_data = "data_analytics/data/devset.csv"
	df = readData(train_data)
	mr_dict = generateDict(df)
	
	test_mr = "name[Adios], food[Spanish], customer rating[high], familyFriendly[yes], near[Ranch]"
	test_slot_attrib_value = getSlotValuePairs(test_mr)
	print(retrieveResponse(mr_dict, test_slot_attrib_value))

	test_df = readData(dev_data)
	generated_utterances = []
	for index, row in test_df.iterrows():
		mr = row['mr']
		utterance = row['ref']

		test_slot_attrib_value = getSlotValuePairs(mr)
		generated_utterances.append(retrieveResponse(mr_dict, test_slot_attrib_value))

	with open('baseline_predictions.txt', mode='w') as f:
		for line in generated_utterances:
			f.write(str(line).replace('"', "")+'\n')


	#test_df["generated_utterance"] = generated_utterances
	#test_df.to_csv("baseline_"+dev_data.split(".")[0]+"-new.csv")
	

	#generated = nltk.word_tokenize("The Blue Spice is a non-family-friendly coffee shop in the city centre near Avalon with a 5 out of 5 customer rating and has a cheap price range.")
	#BLEU: 0.2767774344774222

	#generated = nltk.word_tokenize("It is a nice shop.")
	#ref = [nltk.word_tokenize("Blue Spice as a coffee shop  near Avalon in the city centre may not be a smart choice for family and is less than 20 with a 5 out of 5 rating.")]
	#print(sentence_bleu(ref, generated))

	'''
	result_df = readData("devset-new.csv")
	total_bleu = 0.0
	for index, row in result_df.iterrows():
		ref = [nltk.word_tokenize(row['ref'])]
		generated = nltk.word_tokenize(row['generated_utterance'])
		total_bleu = total_bleu + sentence_bleu(ref, generated)
	avg_bleu = total_bleu/(index+1)
	print("Average BLEU: "+str(avg_bleu))
	'''

	#Average BLEU: 0.234019592971

	







	
