import sys
import json
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess


def create_embeddings(file_paths, path_to_embeddings, path_to_vocab, path_to_model, **params):
    class SentenceGenerator(object):
        def __init__(self, file_paths):
            self.file_paths = file_paths

        def __iter__(self):
            for file_path in self.file_paths:
                for line in open(file_path):
                    # tokenize
                    yield simple_preprocess(line)

    sentences = SentenceGenerator(file_paths)

    model = Word2Vec(sentences, **params)
    model.wv.save_word2vec_format(path_to_model)
    weights = model.wv.syn0
    np.save(open(path_to_embeddings, 'wb'), weights)

    vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    with open(path_to_vocab, 'w') as f:
        f.write(json.dumps(vocab))
