import tensorflow as tf
import data_utils

batch_size = 128
lr = 0.001
epoch_num = 2000
display_step = 10
hidden_size = 128

graph = tf.Graph()
with graph.as_default():
    # get dataset
    d = data_utils.get_files('./data/spoken_numbers_pcm', 2)
    arr_x, arr_y = data_utils.get_dataset(d, 1)
    # training set
    data_x_train = tf.data.Dataset.from_tensor_slices(arr_x[128:])
    data_y_train = tf.data.Dataset.from_tensor_slices(arr_y[128:])
    training_set = tf.data.Dataset.zip((data_x_train, data_y_train)).batch(batch_size).shuffle(512)
    iterator_train = training_set.make_initializable_iterator()
    ne_train = iterator_train.get_next()

    x = tf.placeholder(tf.float32, shape=[None, 168, 13], name='x')
    sparse_index = tf.placeholder(dtype=tf.int64)
    sparse_value = tf.placeholder(dtype=tf.int32)
    sparse_dim = tf.placeholder(dtype=tf.int64)
    sequence_length = tf.placeholder(dtype=tf.int32)
    target = tf.SparseTensor(sparse_index, sparse_value, sparse_dim)

    with tf.name_scope('rnn'):
        rnn_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, num_proj=27)
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
    sess.run(iterator_train.initializer)
    epoch = 1
    step = 1
    while epoch <= epoch_num:
        try:
            data = sess.run(ne_train)
            audio = data[0]
            mfcc = list(map(data_utils.process_ctc, audio))
            idx, val, dim = data_utils.batch2sparse(data[1])
            seq_len = [len(mfcc[0]) for _ in range(len(mfcc))]
            _, l, err, p = sess.run([opt, loss, error, pred], {sparse_index: idx,
                                                               sparse_value: val,
                                                               sparse_dim: dim,
                                                               x: mfcc,
                                                               sequence_length: seq_len
                                                               })
            print("epoch:{:>4}, step:{:>4}, loss:{:>8.4f}, edit_distance:{:>6.2f}".format(epoch, step, l, err))
            step += 1
            if step % 50 == 1:
                print(p)
        except tf.errors.OutOfRangeError:
            sess.run(iterator_train.initializer)
            epoch += 1
