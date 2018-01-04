# use mfcc as features and use ctc to decode the phonetic labels
# use TIMIT dataset

import tensorflow as tf
import data_utils
import pickle
import os

data_path = './data/TIMIT/TRAIN/'
dr_path = list(map(lambda _: os.path.join(data_path, _), os.listdir(data_path)))
training_set = data_utils.get_dataset_timit(dr_path, './data/sentence.pkl')

# path to chr2idx dictionary
chr2idx_dict_path = './data/sentence.pkl'

with open(chr2idx_dict_path, 'rb') as f:
    chr2idx, idx2chr, sentence_dict = pickle.load(f)

lr = 0.001
epoch_num = 1000
display_step = 10
hidden_size = 512
n_classes = len(chr2idx) + 1
batchsize = 128
mfcc_feature_num = 13

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(shape=[None, None, mfcc_feature_num], dtype=tf.float32)
    target = tf.sparse_placeholder(dtype=tf.int32)
    sequence_length = tf.placeholder(shape=(None,), dtype=tf.int32)

    with tf.name_scope('rnn'):
        cell = [tf.nn.rnn_cell.LSTMCell(hidden_size),
                tf.nn.rnn_cell.LSTMCell(hidden_size, num_proj=n_classes)]
        rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cell)
        initial_state = rnn_cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
        outputs, states = tf.nn.dynamic_rnn(rnn_cell, x,
                                            initial_state=initial_state,
                                            dtype=tf.float32)

        logits = outputs

    with tf.name_scope("cal_loss"):
        loss = tf.reduce_mean(
            tf.nn.ctc_loss(labels=target, inputs=logits, sequence_length=sequence_length,
                           time_major=False))

    with tf.name_scope("pred"):
        pred = tf.nn.ctc_beam_search_decoder(tf.transpose(logits, perm=[1, 0, 2]),
                                             sequence_length=sequence_length,
                                             beam_width=100)[0][0]
        error = tf.reduce_mean(tf.edit_distance(pred, tf.to_int64(target), normalize=False))

    opt = tf.train.AdamOptimizer(lr).minimize(loss)

with tf.Session(graph=graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    epoch = 1
    step = 1
    while epoch <= epoch_num:
        for data in data_utils.get_batch(batchsize, training_set[0], training_set[1], training_set[2]):
            _, l, err = sess.run([opt, loss, error], {
                x: data[0],
                target: data[1],
                sequence_length: data[2]
            })
            print("epoch:{:>4}, step:{:>4}, loss:{:>10.4f}, edit_distance:{:>8.2f}".format(epoch, step, l, err))
            if step % display_step == 0:
                p = sess.run(pred, {
                    x: data[0],
                    target: data[1],
                    sequence_length: data[2]
                })
                # convert prediction result from sparse tensor to dense tensor
                dense_pred = sess.run(tf.sparse_to_dense(p[0], p[2], p[1]))
                dense_target = sess.run(tf.sparse_to_dense(data[1][0], data[1][2], data[1][1]))
                print('original:{}'.format(data_utils.to_sentence(dense_target[0], idx2chr)))
                print('predicted:{}'.format(data_utils.to_sentence(dense_pred[0], idx2chr)))
            step += 1
        epoch += 1
