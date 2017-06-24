import sys
import os
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, RepeatVector, Dense, Activation, Input, Flatten, Reshape, Permute, Lambda
from keras.layers.merge import multiply, concatenate
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.callbacks import ModelCheckpoint
#import seq2seq
#from seq2seq.models import AttentionSeq2Seq

import data_loader
import postprocessing


def main():
    path_to_data_dir = 'data/'
    path_to_embeddings_dir = 'embeddings/'

    use_pretrained_embeddings = True        # set to True to use a pre-trained word embedding model
    split_mrs = True                        # set to True to split the test MRs before predicting
    postprocess = True                      # set to False to skip the utterance post-processing
    max_input_seq_len = 30                  # number of words the MRs should be truncated/padded to
    max_output_seq_len = 50                 # number of words the utterances should be truncated/padded to
    vocab_size = 10000                      # maximum vocabulary size of the utterances
    num_variations = 3                      # number of MR permutations to consider for re-ranking
    depth_enc = 1                           # number of LSTM layers in the encoder
    depth_dec = 1                           # number of LSTM layers in the decoder
    hidden_layer_size = 500                 # number of neurons in a single LSTM layer


    # ---- WORD EMBEDDING ----
    print('\nLoading embedding model...')
    embedding_model = data_loader.load_embedding_model(path_to_data_dir, path_to_embeddings_dir, use_pretrained_embeddings)
    weights = embedding_model.syn0

    # DEBUG PRINT
    print('weights.shape =', weights.shape)
    #print(embedding_model.similarity('pizza', 'hamburger'))
    #print(embedding_model.similarity('pizza', 'furniture'))
    #print(embedding_model.similarity('coffee', 'tea'))


    # ---- LOAD DATA ----
    print('\nLoading data...')
    #word2idx, idx2word = data_loader.load_vocab(path_to_vocab)
    x_train, y_train, x_test, y_test, original_mrs, original_sents, test_groups, y_idx2word = \
            data_loader.load_data(path_to_data_dir, embedding_model, vocab_size, max_input_seq_len, max_output_seq_len, num_variations, split_mrs)

    # x_test, y_test = permute_input(original_mrs, original_sents)

    # DEBUG PRINT
    print('Utterance vocab size:', len(y_idx2word))
    print('x_train.shape =', x_train.shape)
    print('y_train.shape =', y_train.shape)
    print('x_test.shape =', x_test.shape)
    print('y_test.shape =', y_test.shape)


    # ---- BUILD THE MODEL ----
    print('\nBuilding language generation model...')
    #model = Sequential()

    #ret_seq_first_layer = False
    #if depth_enc > 1:
    #    ret_seq_first_layer = True

    ## -- ENCODER --
    ##model.add(Embedding(input_dim=weights.shape[0],
    ##                    output_dim=weights.shape[1],
    ##                    weights=[weights],
    ##                    input_length=max_seq_len,       # can be omitted to process sequences of heterogenous length
    ##                    trainable=False))
    #model.add(Bidirectional(LSTM(units=weights.shape[1],
    #                             dropout=0.2,
    #                             recurrent_dropout=0.2,
    #                             return_sequences=ret_seq_first_layer),
    #                        input_shape=(max_input_seq_len, weights.shape[1])))
    #if depth_enc > 2:
    #    for d in range(depth_enc - 2):
    #        model.add(Bidirectional(LSTM(units=weights.shape[1],
    #                                        dropout=0.2,
    #                                        recurrent_dropout=0.2,
    #                                     return_sequences=True)))
    #if depth_enc > 1:
    #    model.add(Bidirectional(LSTM(units=weights.shape[1],
    #                                    dropout=0.2,
    #                                    recurrent_dropout=0.2,
    #                                 return_sequences=False)))

    ## -- DECODER --
    #model.add(RepeatVector(max_output_seq_len))
    #for d in range(depth_dec):
    #    model.add(LSTM(units=weights.shape[1],
    #                   dropout=0.2,
    #                   recurrent_dropout=0.2,
    #                   return_sequences=True))
    #model.add(TimeDistributed(Dense(len(y_idx2word),
    #                                activation='softmax')))


    # ---- ATTENTION MODEL ----

    input = Input(shape=(max_input_seq_len, weights.shape[1]))

    # -- ENCODER --
    encoder = Bidirectional(LSTM(units=hidden_layer_size,
                                 dropout=0.2,
                                 recurrent_dropout=0.2,
                                 return_sequences=True),
                            merge_mode='concat')(input)

    # -- ATTENTION --
    flattened = Flatten()(encoder)

    attention = []
    for i in range(max_output_seq_len):
        weighted = Dense(max_input_seq_len, activation='softmax')(flattened)
        unfolded = Permute([2, 1])(RepeatVector(hidden_layer_size * 2)(weighted))
        multiplied = multiply([encoder, unfolded])
        summed = Lambda(lambda x: K.sum(x, axis=-2))(multiplied)
        attention.append(Reshape((1, hidden_layer_size * 2))(summed))

    attention_out = concatenate(attention, axis=-2)

    # -- DECODER --
    decoder = LSTM(units=hidden_layer_size,
                   dropout=0.2,
                   recurrent_dropout=0.2,
                   return_sequences=True)(attention_out)

    decoder = Dense(len(y_idx2word),
                    activation='softmax')(decoder)

    model = Model(inputs=input, outputs=decoder)


    # ---- Keras Seq2Seq attention model [https://github.com/farizrahman4u/seq2seq] (not working) ----
    #model = AttentionSeq2Seq(input_dim=weights.shape[1],
    #                         input_length=max_input_seq_len,
    #                         hidden_dim=hidden_layer_size,
    #                         output_length=max_output_seq_len,
    #                         output_dim=len(y_idx2word),
    #                         depth=1)


    # ---- COMPILE ----
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.summary()

    # -- Define Checkpoint--
    #filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    filepath = 'trained_model.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    
    # ---- TRAIN ----
    print('\nTraining...')
    model.fit(x_train, y_train,
              batch_size=64,
              epochs=20,
              callbacks=callbacks_list)
    
    
    # ---- TEST ----
    #print('\nTesting...')
    #score, acc = model.evaluate(x_test, y_test)

    #print()
    #print('-> Test score:', score)
    #print('-> Test accuracy:', acc)


    # ---- PREDICT ----
    print('\nPredicting...')

    # -- SINGLE PREDICTION --
    #prediction_distr = model.predict(np.array([x_test[123]]))       # test MR: name[The Rice Boat], food[Japanese], area[city centre]
    #prediction = np.argmax(prediction_distr, axis=2)                # note: prediction_distr is a 3D array even for a single input to model.predict()
    #utterance = [y_idx2word[idx] for idx in prediction[0] if idx > 0]
    #print(' '.join(utterance))

    # -- BATCH PREDICTION --
    results = []
    prediction_distr = model.predict(np.array(x_test))
    predictions = np.argmax(prediction_distr, axis=2)

    for i, prediction in enumerate(predictions):
        utterance = ' '.join([y_idx2word[idx] for idx in prediction if idx > 0])
        results.append(utterance)

    # print(len(original_mrs))
    # print(len(results))
    print("Predictions have been processed. Now we are depermuting them: ")
    # x, y, p = postprocessing.depermute_input(original_mrs, original_sents, results, num_variations)
    # correct_preds = postprocessing.correct(x, p)
    # print(len(original_mrs))
    # print(len(results))
    if split_mrs:
        results_merged = postprocessing.merge_utterances(results, original_mrs, test_groups, num_variations)
    else:
        results_merged = []
        for i, prediction in enumerate(results):
            results_merged.append(postprocessing.relex_utterance(prediction, original_mrs[i]))

    #todo add this
    # if not split_mrs:
    #     utterance = postprocessing.relex_utterance(utterance, original_mrs[i])

    np.savetxt('results/results_raw.txt', list(results_merged), fmt='%s')
    # print('\n'.join(results_merged))


    # ---- POST-PROCESS ----
    if postprocess:
        print("Predictions have been processed. Now we are depermuting them: ")
        x, y, p = postprocessing.depermute_input(original_mrs, original_sents, results_merged, num_variations)
        print("Depermution is done, files written.")
        print("Writing depermute file.")
        cp = postprocessing.combo_print(p, results_merged, num_variations)
        correct_preds = postprocessing.correct(x, p)

        # for pp in p:
        #     print(pp)
        np.savetxt('results/results_pooling.txt', list(p), fmt='%s')
        np.savetxt('results/results_combo_pool.txt', list(cp), fmt='%s')
        np.savetxt('results/results_pooling_corrected.txt', list(correct_preds), fmt='%s')


if __name__ == "__main__":
    sys.exit(int(main() or 0))

    # t = "The Golden Currey serves Fast food food near near"
    # y = "The Golden Currey is rated a 3 3 of a of 5 5 5 5"
    # x = "The Golden Currey is near near the city centre"
    # blah = [t, y, x]
    # mrs = [0,0,0]
    # # t = "The Golden Currey is a family near near"
    # # t = score_grammar_spelling(t, True)
    # tool = language_check.LanguageTool('en-US')
    # for g in blah:
    #     print(score_grammar_spelling(False, g, tool))
    #     print(score_known_errors(g))
    # print(correct(mrs, blah))
