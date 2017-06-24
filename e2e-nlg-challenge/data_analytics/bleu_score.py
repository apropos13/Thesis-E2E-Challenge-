import subprocess
import sys
import ast
import re
import nltk
    
def get_blue(filename_utt,filename_pred):
	
	cmd='perl multi-bleu.perl '+filename_utt+' < '+ filename_pred
	perl_bleu4 = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE).communicate()[0].split()[2].split('/')[3]
	#perl_bleu4 = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE).communicate()[0]
	return float(perl_bleu4)
	#to get bleu 1,2,3 simply change [3]-->[1] or [2] or [3]
	
if __name__ == '__main__':
	

	print("Blue for char LSTM")
	utt='../lstm/data/devset.txt'
	pred='../lstm/CharModelOutput/char_out.txt'
	print(get_blue(utt,pred))



