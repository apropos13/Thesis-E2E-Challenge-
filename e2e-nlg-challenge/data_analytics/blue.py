import pandas as pd
import nltk

######SPECIFY PARAMETERS#############
filename_test='data/devset.csv'
#filename_predictions='dummy_output_file.txt'
#####################################

def mean(numbers):
	print("TOTAL elems=" +str(len(numbers)))
	return float(sum(numbers)) / len(numbers)

def get_blue(filename_test, filename_predictions):
	df=pd.read_csv(filename_test)
	#df=pd.read_csv('data/trainset.csv', encoding="latin-1") #use this if you get encoding erros
	df['mr']=df['mr'].astype('str')
	df['ref']=df['ref'].astype('str')
	ref_df=df['ref']

	ref_list=[] # list to store references
	for row in ref_df:
		ref_list.append(row.split())

	prediction_list=[] #list to store predictions
	with open(filename_predictions) as f:
		for utterance in f:
			prediction_list.append(utterance.split())

	#sanity check
	print len(prediction_list)
	print len(ref_list)
	assert len(prediction_list)==len(ref_list), "Number of predictions is not equal to the number of references"

	print("###### File: "+ str(filename_predictions)+"#######")
	print("Getting Blue corpus score...")
	BLEU_coprus= nltk.translate.bleu_score.corpus_bleu(ref_list, prediction_list)
	print "Blue (corpus)=", BLEU_coprus

	#also calculate the mean BLUE score of each prediction-realization pair
	
	#print("Getting Blue sentence score...")
	#BLEU_sentece= mean([nltk.translate.bleu_score.sentence_bleu(ref_list[x], prediction_list[x]) for x in range(len(ref_list))])
	#print "Blue (sentence)=", BLEU_sentece


	print("#############")

if __name__ == '__main__':
	
	#print("MR SPLITTING:")
	#get_blue(filename_test, 'MR_SplittingResults/results_mr_splitting_10_epochs.txt')
	#get_blue(filename_test, 'MR_SplittingResults/results_mr_splitting_20_epochs.txt')
	#get_blue(filename_test, 'MR_SplittingResults/results_mr_splitting_layers_2_40_epochs.txt')

	#print("LATEST RESULTS")
	#get_blue(filename_test, 'FinalResults/results_corrected_3_slot_mrs_30_epochs.txt')
	#get_blue(filename_test, 'FinalResults/results_corrected_15_epochs.txt')

	#print("BASELINE")
	#get_blue(filename_test, 'FinalResults/baseline_predictions.txt')
#

	get_blue(filename_test, '../lstm/CharModelOutput/char_out.txt')


	#print("Initial")
	#get_blue(filename_test, 'FinalResults/initial_result.txt')
#
	#print("Initial+Delex")
	#get_blue(filename_test, 'FinalResults/delexed.rtf')
#
	#print("Initial+Delex+MR Splitting")
	#get_blue(filename_test, 'FinalResults/results_corrected_15_epochs.txt')

	
