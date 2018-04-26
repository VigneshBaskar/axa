# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# <codecell>

from Scripts.UnknownWordsProcessing import UnknownWordsProcessing 
from Scripts.VocabDict import VocabDict
from Scripts.MapWordToID import MapWordToID
from Scripts.Tokenizer import word_tokenizer
from Scripts.SentenceProcessing import SentenceProcessing
from Scripts.Word2VecUtilities import Word2VecUtilities

# <codecell>

with open(os.path.join('Data','data.p'), 'rb') as handle:
    data = pickle.load(handle)

# <codecell>

X_text = data['X_text']
y = data['y']

# <codecell>

all_documents_tokenized_words = [list(set(word_tokenizer(text))) for text in X_text]
vocab_dict, rev_vocab_dict = VocabDict.create_vocab_dict(all_documents_tokenized_words, min_doc_count=1000)

# <codecell>

unknown_words_processing = UnknownWordsProcessing(vocab_dict.keys(), replace=False)
tokenized_documents = [word_tokenizer(text) for text in X_text]
unknown_words_removed_sentences = unknown_words_processing.remove_or_replace_unkown_word_from_sentences(tokenized_documents)
preprocessed_documents = SentenceProcessing().pad_truncate_sent(unknown_words_removed_sentences, chosen_sent_len = 300)

# <codecell>

w2v_model = Word2VecUtilities.create_word2vector_model(unknown_words_removed_sentences, wv_size=50)
embedding_matrix = Word2VecUtilities.create_embeddings_matrix(w2v_model, rev_vocab_dict)

# <codecell>

vocab_dict['my_dummy']=len(vocab_dict)
rev_vocab_dict[len(rev_vocab_dict)] = 'my_dummy'
embedding_matrix = np.vstack((embedding_matrix, np.zeros((1, embedding_matrix.shape[1]))))

# <codecell>

map_word_to_id = MapWordToID(vocab_dict)
id_lists = map_word_to_id.word_lists_to_id_lists(preprocessed_documents)
id_arrays = np.array(id_lists)

# <codecell>

def return_actual_text(x, rev_vocab_dict):
    actual_text = " ".join([rev_vocab_dict[word_id] for word_id in x])
    return actual_text

# <codecell>

w2v_model.wv.most_similar('grant')

# <codecell>

X_train_and_valid, X_test, y_train_and_valid, y_test = train_test_split(id_arrays, y, test_size=0.15, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_and_valid, y_train_and_valid, test_size=0.15, random_state=42)

# <codecell>

data_X_y = {'X_train':X_train, 'X_valid':X_valid, 'X_test':X_test,
           'y_train':y_train, 'y_valid':y_valid,'y_test':y_test}

with open(os.path.join('Data','data_X_y.p'), 'wb') as handle:
    pickle.dump(data_X_y, handle, protocol=pickle.HIGHEST_PROTOCOL)

# <codecell>

training_params = {'embedding_matrix':embedding_matrix, 'vocab_size':len(vocab_dict)}
with open(os.path.join('Data','training_params.p'), 'wb') as handle:
    pickle.dump(training_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# <codecell>


