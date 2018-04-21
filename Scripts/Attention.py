import tensorflow as tf
from tensorflow.contrib.layers import fully_connected


def create_attention(conc_outputs, attention_n_neurons):
    # conc_outputs : (batch_size, n_time_steps, rnn_output_size)
    # attention_n_neurons : attention_network size
    # attention_output : (batch_size, rnn_output_size)
    # normalized_attention_scores : (batch_size, n_time_steps, 1)
    
    attention_vector = tf.get_variable("attention_vector", shape=(attention_n_neurons, 1),
                                         initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

    conc_outputs_reshaped = tf.reshape(conc_outputs, shape=(-1, conc_outputs.get_shape().as_list()[-1]),
                                         name='conc_outputs_reshaped')
    
    conc_outputs_reshaped_fully_connected = fully_connected(conc_outputs_reshaped, attention_n_neurons,
                                                              activation_fn=tf.nn.tanh,
                                                            scope='conc_outputs_reshaped_fully_connected')

    un_normalized_attention_scores = tf.matmul(conc_outputs_reshaped_fully_connected, attention_vector,
                                               name='un_normalized_attention_scores')
    
    un_normalized_attention_scores_reshaped = tf.reshape(un_normalized_attention_scores,
                                                            shape=(-1, conc_outputs.get_shape().as_list()[1]))
    
    normalized_attention_scores = tf.expand_dims(tf.nn.softmax(un_normalized_attention_scores_reshaped),2)

    attentive_conc_outputs = tf.multiply(normalized_attention_scores, conc_outputs)
    attention_output  = tf.reduce_sum(attentive_conc_outputs, axis=1)
    return attention_output, normalized_attention_scores