import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout 
from keras.layers import LSTM, RepeatVector, Dense, Activation
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
import pandas as pd
from string import printable

def pretty_format(sample):
	charIds = np.zeros(sample.shape[0])

	for (idx, elem) in enumerate(sample):
		charIds[idx] =np.nonzero(elem)[0].squeeze()
	return  ''.join(np.array([ int_to_char[x] for x in charIds])) 

def get_sequences(dataframe, max_sequen_len, dict_len):
	x_seq=np.zeros( (dataframe.shape[0], max_sequen_len, dict_len), dtype=np.bool)
	y_seq=np.zeros( (dataframe.shape[0], max_sequen_len, dict_len), dtype=np.bool)
	#iterators
	i=0
	for index, row in dataframe.iterrows():
		x_seq[i,0, char_to_int['S']]=1
		y_seq[i,0, char_to_int['S']]=1
		for j in range(1, max_seq_length):
			if j< len(row['mr'])+1:
				try:
					x_seq[i,j, char_to_int[row['mr'][j-1] ] ]=1
				except KeyError:
					#pass 
					x_seq[i,j, char_to_int[ ' ' ] ]=1 #CHANGE THIS!!!
			else:
				x_seq[i,j, char_to_int['E']]=1 #end of str

			if j< len(row['ref'])+1:
				try:
					y_seq[i,j,char_to_int[row['ref'][j-1] ] ]=1
				except KeyError:
					y_seq[i,j,char_to_int[' ']]=1 #CHANGE THIS!!!
			else:
				y_seq[i,j, char_to_int['E']]=1 #end of str

			
		i+=1

	return (x_seq,y_seq)




if __name__ == '__main__':
	

	filename_train='data/devset_3_slot_mr.csv' #set to traiset
	#df_train=pd.read_csv(filename_train, encoding="latin-1") #linux/windows
	df_train=pd.read_csv(filename_train) #mac
	filename_test='data/devset_3_slot_mr.csv'
	#df_test=pd.read_csv(filename_test,encoding="latin-1")
	df_test=pd.read_csv(filename_test)



	df_train['mr']=df_train['mr'].astype('str').str.lower() #store column
	df_train['ref']=df_train['ref'].astype('str').str.lower()

	df_test['mr']=df_test['mr'].astype('str').str.lower() 
	df_test['ref']=df_test['ref'].astype('str').str.lower() 
	



	####set parameters#####
	mr_size=100 				#char_size of each mr
	max_output_seq_len = 1+max(len(r['ref']) for i,r in df_train.iterrows() )	

	max_input_seq_len = 1+max(len(r['mr']) for i,r in df_train.iterrows() )

	depth_enc = 1               # number of LSTM layers in the encoder
	depth_dec = 1 
	hiddenStateSize = 187
	hiddenLayerSize = 187          
	#######################
	st = set(printable.lower()) #vocabulary
	chars= sorted(list(st)) #vocabulary
	char_to_int= dict((c,i) for i,c in enumerate(chars))
	int_to_char = dict((i, c) for i, c in enumerate(chars))



	#add a special starting and ending character to the dictionary with value start_index.
	start_index=len(char_to_int)
	char_to_int['S']=start_index 
	int_to_char[start_index]='S'
	char_to_int['E']=start_index+1
	int_to_char[start_index+1]='E'


	#print summary statistics ( str.len() is vectorized in pandas)
	n_chars_train=df_train['mr'].str.len().sum()+ df_train['ref'].str.len().sum()
	n_chars_test=df_test['mr'].str.len().sum()+ df_test['ref'].str.len().sum()
	n_vocab=len(chars)
	#message
	print ("Number of characters in train (before cleanup)="+str(n_chars_train))

	print ("Number of characters in trest (before cleanup)="+str(n_chars_test))
	print ("Size of vocabulary ="+str(n_vocab))



	#prepare the dataset of the input to output pairs encoded as integers
	max_seq_length=max(max_input_seq_len,max_output_seq_len)
	print ("MRs will be padded/truncated to size ="+str(max_seq_length))

	print("Creating Input Sequences...")
	x_train,y_train=get_sequences(df_train, max_seq_length, len(char_to_int))
	print("Creating Output Sequences...")
	x_test, y_test=get_sequences(df_test, max_seq_length, len(char_to_int))


	#DEBUG PRINT
	train_x=x_train[10,:,:]
	print("Train sample input: \n" + pretty_format(train_x) )
	train_y=y_train[10,:,:]
	print("Train sample output: \n" + pretty_format(train_y) )

	test_x=x_test[10,:,:]
	print("Test sample input: \n" + pretty_format(test_x) )
	test_y=y_test[10,:,:]
	print("Test sample output: \n" + pretty_format(test_y) )

	

	# ---- BUILD THE MODEL ----
	print('\nBuilding language generation model...')
	model = Sequential()

	# -- ENCODER --
	model.add(Bidirectional(LSTM(units=hiddenLayerSize,
                                 dropout=0.2,
                                 recurrent_dropout=0.2,
                                 return_sequences=False),
                            input_shape=(max_seq_length, len(char_to_int))))

	# -- DECODER --
	model.add(RepeatVector(max_output_seq_len))
	for d in range(depth_dec):
		model.add(LSTM(units=hiddenLayerSize,
			dropout=0.2,
			recurrent_dropout=0.2,
			return_sequences=True))
		model.add(TimeDistributed(Dense(len(char_to_int),
			activation='softmax')))

	model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001))
	print (model.summary())

	# Test a simple prediction on a batch for this model.
	print("Sample input Batch size:"),
	print(x_train[0:32, :, :].shape)
	print("Sample input Batch labels (y_train):"),
	print(y_train[0:32, :, :].shape)
	outputs = model.predict(y_train[0:32, :, :])
	print("Output Sequence size:"),
	print(outputs.shape)

	# ---- Define Checkpoint----
	filepath="CharmodelWeights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf"
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]

	# ---- TRAIN ----
	print('\nTraining...')
	model.fit(x_train,
	 y_train, 
	 batch_size = 128, 
	 epochs = 1,
	 callbacks=callbacks_list)

	total=''
	print("Predicting on all test data...")

	
	for mr_id in range(0,x_test.shape[0]):

		# Test a simple prediction on a batch for this model.
		inputMr = x_test[mr_id:mr_id+1, :, :]
		outputs = model.predict(inputMr)

		sample_in=inputMr[0]
		mr=pretty_format(sample_in)
		
		utt= ''.join([int_to_char[x.argmax()] for x in outputs[0, :, :]])

		total= total+'\n \n'+ mr+'\n'+utt

	print("Saving output to text file...")
	with open('char_out.txt','w') as f:
		f.write(total)
