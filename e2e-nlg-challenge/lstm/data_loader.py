import os
import json
import pickle
import copy
import random
import re
import pandas as pd
import numpy as np
from nltk import FreqDist
from gensim.models import Word2Vec, KeyedVectors
from keras.utils import to_categorical
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

import embedding

# TODO: rewrite into object-oriented

def load_embedding_model(path_to_data_dir, path_to_embeddings_dir, use_pretrained_embeddings):
    path_to_training = path_to_data_dir + 'trainset_perm_3_slot_mr.csv'
    path_to_test = path_to_data_dir + 'devset_3_slot_mr.csv'
    path_to_embeddings = path_to_embeddings_dir + 'embeddings.npy'
    path_to_vocab = path_to_embeddings_dir + 'vocab.json'
    path_to_model = path_to_embeddings_dir + 'embedding_model.bin'
    #path_to_pretrained_model = path_to_embeddings_dir + 'GoogleNews-vectors-negative300.bin'
    path_to_pretrained_model = path_to_embeddings_dir + 'glove.6B.300d.txt'


    if use_pretrained_embeddings:
        # load Google's word2vec pre-trained word embedding model
        #return KeyedVectors.load_word2vec_format(path_to_pretrained_model, binary=True)

        # load Stanford's GloVe pre-trained word embedding model
        return KeyedVectors.load_word2vec_format(path_to_pretrained_model, binary=False)
    else:
        # train custom embedding model, if necessary
        if (os.path.isdir(path_to_embeddings_dir) == False or os.path.isfile(path_to_embeddings) == False):
            embedding.create_embeddings([path_to_training, path_to_test],
                                        path_to_embeddings,
                                        path_to_vocab,
                                        path_to_model,
                                        size=100,
                                        min_count=2,
                                        window=5,
                                        iter=1)

        # load our trained word2vec model
        return KeyedVectors.load_word2vec_format(path_to_model, binary=False)


def load_data(path_to_data_dir, embedding_model, vocab_size, max_input_seq_len, max_output_seq_len, num_variations, split_mrs):
    path_to_training = path_to_data_dir + 'trainset.csv'
    #path_to_training = path_to_data_dir + 'trainset_perm_3_slot_mr.csv'
    path_to_test = path_to_data_dir + 'devset.csv'
    #path_to_test = path_to_data_dir + 'devset_3_slot_mr.csv'
    #path_to_data_embed = path_to_data_dir + 'data_embed.pkl'
    

    # store/load the data in the embedded form
    #if os.path.isfile(path_to_data_embed) == False:
    #    x_train, y_train, x_test, y_test = preprocess_data(path_to_training, path_to_test, embedding_model, max_seq_len)
    #    with open(path_to_data_embed, 'wb') as f:
    #        pickle.dump([x_train, y_train, x_test, y_test], f)
    #else:
    #    with open(path_to_data_embed, 'rb') as f:
    #        x_train, y_train, x_test, y_test = pickle.load(f)

    #return x_train, y_train, x_test, y_test

    return preprocess_data(path_to_training, path_to_test, embedding_model, vocab_size, max_input_seq_len, max_output_seq_len, num_variations, split_mrs)


def preprocess_data(path_to_training_data, path_to_test_data, embedding, vocab_size, max_input_seq_len, max_output_seq_len, num_variations, use_split_mrs):
    # read the training data from file
    data_frame_train = pd.read_csv(path_to_training_data, header=0, encoding='latin1')  # names=['mr', 'ref']
    x_train = data_frame_train.mr.tolist()
    y_train = data_frame_train.ref.tolist()

    # read the test data from file
    data_frame_test = pd.read_csv(path_to_test_data, header=0, encoding='latin1')       # names=['mr', 'ref']
    x_test = data_frame_test.mr.tolist()
    y_test = data_frame_test.ref.tolist()

    original_mrs = copy.deepcopy(x_test)
    original_sents = copy.deepcopy(y_test)

    if use_split_mrs:
        # split MRs into shorter ones
        x_test, y_test, test_groups = split_mrs(x_test, y_test, num_variations=num_variations)
    elif num_variations > 1:
        x_test, y_test = permute_input(x_test, y_test, num_permutes=num_variations)
        test_groups = []
    else:
        test_groups = []


    # parse the utterances into lists of words
    y_train = [preprocess_utterance(y) for y in y_train]
    y_test = [preprocess_utterance(y) for y in y_test]

    # create utterance vocabulary
    distr = FreqDist(np.concatenate(y_train + y_test))
    y_vocab = distr.most_common(min(len(distr), vocab_size))        # cap the vocabulary size
    y_idx2word = [word[0] for word in y_vocab]
    y_idx2word.insert(0, '-PADDING-')
    y_idx2word.extend(['&slot_val_name&', '&slot_val_food&', '&slot_val_near&'])
    y_idx2word.append('-PERIOD-')
    y_idx2word.append('-NA-')
    y_word2idx = {word: idx for idx, word in enumerate(y_idx2word)}

    delex_data(x_train, y_train, update_data_source=True)
    delex_data(x_test, y_test, update_data_source=True)
    

    padding_vec = np.zeros(embedding.syn0.shape[1])         # embedding vector for "padding" words

    # produce sequences of embedding vectors from the meaning representations (MRs) in the training set
    x_train_seq = []
    for mr in x_train:
        row_list = []
        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            # parse the slot and convert to embedding
            slot = slot_value[:sep_idx].strip()
            row_list.extend([embedding[slot_word] for slot_word in slot.split() if slot_word in embedding.vocab])
            # parse the value and convert to embedding
            value = slot_value[sep_idx + 1:-1].strip()
            row_list.extend([embedding[value_word] for value_word in value.split() if value_word in embedding.vocab])
        # add padding
        row_list = add_padding(row_list, padding_vec, max_input_seq_len)

        x_train_seq.append(row_list)

    # produce sequences of one-hot vectors from the reference utterances in the training set
    y_train_seq = np.zeros((len(y_train), max_output_seq_len, len(y_word2idx)), dtype=np.int8)
    for i, utterance in enumerate(y_train):
        for j, word in enumerate(utterance):
            # truncate long utterances
            if j >= max_output_seq_len:
                break

            # represent each word with a one-hot vector
            if word == '.':
                y_train_seq[i][j][y_word2idx['-PERIOD-']] = 1
            elif word in y_word2idx:
                y_train_seq[i][j][y_word2idx[word]] = 1
            else:
                y_train_seq[i][j][y_word2idx['-NA-']] = 1

        # add padding for short utterances
        for j in range(len(utterance), max_output_seq_len):
            y_train_seq[i][j][y_word2idx['-PADDING-']] = 1

    # produce sequences of embedding vectors from the meaning representations (MRs) in the test set
    x_test_seq = []
    for mr in x_test:
        row_list = []
        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            # parse the slot and convert to embedding
            slot = slot_value[:sep_idx].strip()
            row_list.extend([embedding[slot_word] for slot_word in slot.split() if slot_word in embedding.vocab])
            # parse the value and convert to embedding
            value = slot_value[sep_idx + 1:-1].strip()
            row_list.extend([embedding[value_word] for value_word in value.split() if value_word in embedding.vocab])
        # add padding
        row_list = add_padding(row_list, padding_vec, max_input_seq_len)

        x_test_seq.append(row_list)

    # produce sequences of one-hot vectors from the reference utterances in the test set
    y_test_seq = np.zeros((len(y_test), max_output_seq_len, len(y_word2idx)), dtype=np.int8)
    for i, utterance in enumerate(y_test):
        for j, word in enumerate(utterance):
            # truncate long utterances
            if j >= max_output_seq_len:
                break

            # represent each word with a one-hot vector
            if word in y_word2idx:
                y_test_seq[i][j][y_word2idx[word]] = 1
            else:
                y_test_seq[i][j][y_word2idx['-NA-']] = 1

        # add padding for short utterances
        for j in range(len(utterance), max_output_seq_len):
            y_test_seq[i][j][y_word2idx['-PADDING-']] = 1

    return (np.array(x_train_seq), np.array(y_train_seq), np.array(x_test_seq), np.array(y_test_seq), original_mrs, original_sents, test_groups, y_idx2word)


def permute_input(mrs, sents, num_permutes):
    new_mr = []
    new_sent = []
    for x, mr in enumerate(mrs):
        sentence = sents[x]
        temp = []
        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            slot = slot_value[:sep_idx].strip()
            value = slot_value[sep_idx + 1:-1].strip()
            temp.append(slot + '[' + value + ']')
        for t in range(0, num_permutes):
            temptemp = copy.deepcopy(temp)
            random.shuffle(temptemp)
            curr_mr = ', '.join(temptemp)
            new_mr.append(curr_mr)
            new_sent.append(sentence)
    return new_mr, new_sent


def split_mrs(mrs, utterances, num_variations):
    new_mrs = []
    new_utterances = []
    groups = []
    group_id = 0

    for idx, mr in enumerate(mrs):
        utterance = utterances[idx]
        # do not split short MRs
        if len(mr) < 4:
            new_mrs.append(mr)
            new_utterances.append(utterance)
            continue

        slot_value_list = []
        name_slot = ()

        # parse the slot-value pairs
        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            slot = slot_value[:sep_idx].strip()
            value = slot_value[sep_idx + 1:-1].strip()

            if slot == 'name':
                name_slot = (slot, value)
            else:
                slot_value_list.append((slot, value))

        for i in range(num_variations):
            slot_value_list_copy = slot_value_list[:]
            random.shuffle(slot_value_list_copy)

            # distribute the slot-value pairs as multiple shorter MRs
            while len(slot_value_list_copy) > 0:
                # include the name slot by default in each subset
                mr_subset = [name_slot]
                # add up to two other slots to the subset
                for i in range(min(2, len(slot_value_list_copy))):
                    mr_subset.append(slot_value_list_copy.pop())
            
                new_mr = [s + '[' + v + ']' for s, v in mr_subset]
                new_mrs.append(', '.join(new_mr))
                new_utterances.append(utterance)
                groups.append(group_id)
            
            group_id += 1

    return new_mrs, new_utterances, groups


def preprocess_utterance(utterance, keep_periods=False):
    if keep_periods:
        chars_to_filter = '!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n'
    
        # add spaces before periods so they can be parsed as individual words
        utterance = utterance.replace('. ', ' . ')
        if utterance[-1] == '.':
            utterance = utterance[:-1] + ' ' + utterance[-1]

        return text_to_word_sequence(utterance, filters=chars_to_filter)
    else:
        chars_to_filter = '.!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n'

        return text_to_word_sequence(utterance, filters=chars_to_filter)


def delex_data(mrs, sentences, update_data_source=False, specific_slots=None, split=True):
    if specific_slots is not None:
        delex_slots = specific_slots
    else:
        delex_slots = ['name', 'food', 'near']

    for x, mr in enumerate(mrs):
        if split:
            sentence = ' '.join(sentences[x])
        else:
            sentence = sentences[x].lower()
        for slot_value in mr.split(','):
            sep_idx = slot_value.find('[')
            # parse the slot
            slot = slot_value[:sep_idx].strip()
            if slot in delex_slots:
                value = slot_value[sep_idx + 1:-1].strip()
                sentence = sentence.replace(value.lower(), '&slot_val_{0}&'.format(slot))
                mr = mr.replace(value, '&slot_val_{0}&'.format(slot))
                # if not split:
                #     print("delex:")
                #     print('&slot_val_{0}&'.format(slot))
                #     print(value.lower())
                #     print(sentence)
        if update_data_source:
            if split:
                sentences[x] = sentence.split()
            else:
                sentences[x] = sentence
            mrs[x] = mr
        if not split:
            return sentence
        # new_sent = relex_sentences(mr, sentence)


def add_padding(seq, padding_vec, max_seq_len):
    diff = max_seq_len - len(seq)
    if diff > 0:
        # pad short sequences
        return seq + [padding_vec for i in range(diff)]
    else:
        # truncate long sequences
        return seq[:max_seq_len]


def load_vocab(path_to_vocab):
    with open(path_to_vocab, 'r') as f_vocab:
        data = json.loads(f_vocab.read())

    word2idx = data
    idx2word = {v: k for k, v in data.items()}

    return word2idx, idx2word
