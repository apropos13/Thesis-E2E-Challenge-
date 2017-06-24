import pandas, csv


def readData(filename):
	
	df = pandas.read_csv(filename, "utf8", names = ["mr"])
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
			mr.append(str(new_row[0]).replace("\"", "").strip())
			ref.append(str(new_row[1]))
			#generated_utterance.append(str(new_row[2]))
	df_new['mr'] = mr
	df_new['ref'] = ref
	#df_new['generated_utterance'] = generated_utterance
	return(df_new)
	


if __name__ == "__main__":

	frames = []
	for i in range(4, 9):
		filename = "perm_"+str(i)+"_slot_mr.csv"
		new_df = readData(filename)
		frames.append(new_df)

	all_data_df = pandas.concat(frames)
	all_data_df.to_csv("all-perm-data.csv")
