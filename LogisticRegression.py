# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

import os
import pickle
import numpy as np
import tensorflow as tf

# <codecell>

from tensorflow.contrib.layers import fully_connected
from sklearn.preprocessing import MultiLabelBinarizer

# <codecell>

from Scripts.CreateTrainingBatches import CreateTrainingBatches

# <codecell>

with open(os.path.join('Data','data_X_y.p'), 'rb') as handle:
    data_X_y = pickle.load(handle)

with open(os.path.join('Data','training_params.p'), 'rb') as handle:
    training_params = pickle.load(handle)

# <codecell>

X_train, X_valid, X_test = [data_X_y['X_train'],data_X_y['X_valid'], data_X_y['X_test']]
y_train, y_valid, y_test = [data_X_y['y_train'],data_X_y['y_valid'], data_X_y['y_test']]

# <codecell>

vocab_size = training_params['vocab_size']
mlb = MultiLabelBinarizer()
X_valid = mlb.fit_transform(X_valid)
X_train = mlb.fit_transform(X_train)
create_training_batches_object = CreateTrainingBatches(X_train, y_train, X_valid, y_valid)

# <codecell>

def print_metrics(np_prob, np_y):
    neg_accuracy = np.mean((np_prob<0.5)[(np_y==0)])
    pos_accuracy = np.mean((np_prob>0.5)[(np_y==1)])
    accuracy = np.mean((pos_accuracy, neg_accuracy))
    print('Negative accuracy',neg_accuracy)
    print('Positive accuracy',pos_accuracy)
    print('Accuracy', accuracy)
    return accuracy

# <codecell>

tf.reset_default_graph()
doc_vectors = tf.placeholder(dtype=tf.float32,shape=[None, vocab_size], name='doc_vectors')
y = tf.placeholder(tf.float32, [None, 1], name='y')

# <codecell>

learning_rate = 0.01

# <codecell>

layer_one_output = fully_connected(doc_vectors, 100, activation_fn=tf.nn.relu)
logits = fully_connected(layer_one_output,1, activation_fn=None)
prob = tf.nn.sigmoid(logits, name='prob')

x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits, name='x_entropy')
loss = tf.reduce_mean(x_entropy, name='loss')

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss, name='train_op')

# <codecell>

file_writer = tf.summary.FileWriter('tf_logs/logistic_regression', tf.get_default_graph())

# <codecell>

init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.InteractiveSession()
init.run()

# <codecell>

highest_validation_accuracy = 0.5
for i in range(2000):
    x_train_samples, y_train_samples = create_training_batches_object.create_training_data()
    
    _, np_prob, np_y, np_loss = sess.run([training_op, prob, y, loss],
                                              feed_dict={doc_vectors: x_train_samples,
                                                         y: y_train_samples})
    if i%100==0:
        print('Epoch', i, 'Loss',np_loss)
        x_valid_samples, y_valid_samples = create_training_batches_object.create_validation_data()

        np_prob, np_y, np_loss = sess.run([prob, y, loss],
                                                  feed_dict={doc_vectors: x_valid_samples,
                                                             y: y_valid_samples})
        validation_accuracy = print_metrics(np_prob, np_y)
        if validation_accuracy > highest_validation_accuracy:
            print('Saved model with highest accuracy')
            saver.save(sess, os.path.join('Models', 'tf_models','model.ckpt'))
            highest_validation_accuracy = validation_accuracy
        print('-----------------------------')
