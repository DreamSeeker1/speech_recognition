import tensorflow as tf
import data_utils
import numpy as np

lr = 0.001
epoch_num = 5000
display_step = 10
hidden_size = 128
n_classes = 40

mfcc, y, seq_len = data_utils.load_npy('./data/sample_data/mfcc', './data/sample_data/char_y')
sparse = data_utils.batch2sparse(y)

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(shape=[None, None, 26], dtype=tf.float32)
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
        # pred = tf.nn.ctc_greedy_decoder(tf.transpose(logits, perm=[1, 0, 2]),
        #                                 sequence_length=sequence_length)[0][0]
        pred = tf.nn.ctc_beam_search_decoder(tf.transpose(logits, perm=[1, 0, 2]),
                                             sequence_length=sequence_length,
                                             beam_width=100)[0][0]
        error = tf.reduce_mean(tf.edit_distance(pred, tf.to_int64(target), normalize=False))

    opt = tf.train.AdamOptimizer(lr).minimize(loss)

with tf.Session(graph=graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    epoch = 1
    while epoch <= epoch_num:
        _, l, err, p = sess.run([opt, loss, error, pred], {
            x: np.array(mfcc),
            target: sparse,
            sequence_length: seq_len
        })
        print("epoch:{:>4}, loss:{:>10.4f}, edit_distance:{:>6.2f}".format(epoch, l, err))
        epoch += 1
