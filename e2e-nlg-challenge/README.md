# e2e-nlg-challenge
LSTM-based encoder-decoder model for the E2E NLG Challenge (http://www.macs.hw.ac.uk/InteractionLab/E2E/).

To use a pre-trained word embedding model, download Stanford's GloVe (http://nlp.stanford.edu/data/glove.6B.zip). After unzipping, move the glove.6B.300d.txt file to the lstm/embeddings folder and insert "367297 300" as the first line (without the quotes); the numbers correspond to the size of the vocabulary and the embedding dimension, respectively. This is necessary for the gensim library to be able to load the embedding model from the file.
An alternative embedding model the program works with is Google's Word2vec (https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit).

Dependencies:
keras
tensorflow (tensorflow-gpu)
gensim
nltk
pandas
numpy
