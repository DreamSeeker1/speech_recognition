# use mfcc as features and use ctc to decode the phonetic labels
# use TIMIT dataset

import tensorflow as tf
import data_utils
import pickle
import os

# path to chr2idx dictionary
chr2idx_dict_path = './data/sentence.pkl'

data_path = './data/TIMIT/TRAIN/'
dr_path = list(map(lambda _: os.path.join(data_path, _), os.listdir(data_path)))
training_set = data_utils.get_dataset_timit(dr_path, chr2idx_dict_path)

with open(chr2idx_dict_path, 'rb') as f:
    chr2idx, idx2chr, sentence_dict = pickle.load(f)

lr = 0.001
epoch_num = 1000
display_step = 10
hidden_size = 128
n_classes = len(chr2idx) + 1
batchsize = 128
mfcc_feature_num = 13
isTrain = True
cell_nums = 2
RNN_Cell = tf.nn.rnn_cell.GRUCell
momentum = 0.99

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(shape=[None, None, mfcc_feature_num], dtype=tf.float32, name='mfcc')
    x_expand = tf.expand_dims(x, 3)
    target = tf.sparse_placeholder(dtype=tf.int32, name='target')
    sequence_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='sequence_length')
    epoch_g = tf.Variable(1, False, name='epoch')
    step_g = tf.Variable(1, False, name='step')
    with tf.name_scope('conv_layer'):
        conv1 = tf.layers.conv2d(x_expand, filters=4, kernel_size=3, padding='same', activation=tf.nn.leaky_relu)
        conv2 = tf.layers.conv2d(conv1, filters=8, kernel_size=3, padding='same', activation=tf.nn.leaky_relu)
        conv_shape = tf.shape(conv2)
        conv_out = tf.reshape(conv2, (conv_shape[0], conv_shape[1], 13 * 8))
    with tf.name_scope('rnn'):
        cell_fw = [RNN_Cell(hidden_size) for _ in range(cell_nums)]
        cell_bw = [RNN_Cell(hidden_size) for _ in range(cell_nums)]
        rnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(cell_fw)
        rnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(cell_bw)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, conv_out,
                                                          sequence_length=sequence_length, dtype=tf.float32)
        logits = tf.concat(outputs, 2)
        logits = tf.layers.dense(logits, n_classes)
    with tf.name_scope("cal_loss"):
        loss = tf.reduce_mean(
            tf.nn.ctc_loss(labels=target, inputs=logits, sequence_length=sequence_length,
                           time_major=False))

    with tf.name_scope("make_prediction"):
        pred = tf.nn.ctc_beam_search_decoder(tf.transpose(logits, perm=[1, 0, 2]),
                                             sequence_length=sequence_length,
                                             beam_width=100)[0][0]
    with tf.name_scope("cal_edit_distance"):
        error = tf.reduce_mean(tf.edit_distance(pred, tf.to_int64(target), normalize=False))

    opt = tf.train.AdamOptimizer(lr).minimize(loss)
    tf.summary.scalar('edit_distance', error)
    tf.summary.scalar('ctc_loss', loss)

with tf.Session(graph=graph) as sess:
    if isTrain:
        writer = tf.summary.FileWriter('./tmp/summary', graph=graph)
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()
        try:
            ckpt_path = tf.train.latest_checkpoint('./tmp/checkpoint/')
            saver.restore(sess, ckpt_path)
        except ValueError:
            init = tf.global_variables_initializer()
            sess.run(init)
        epoch = sess.run(epoch_g)
        while epoch <= epoch_num:
            for data in data_utils.get_batch(batchsize, training_set[0], training_set[1], training_set[2]):
                _, l, err, mgd = sess.run([opt, loss, error, merged], {
                    x: data[0],
                    target: data[1],
                    sequence_length: data[2]
                })
                step = sess.run(step_g)
                writer.add_summary(mgd, global_step=step)
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
                sess.run(step_g.assign_add(1))
            sess.run(epoch_g.assign_add(1))
            epoch = sess.run(epoch_g)
            if epoch % 5 == 0:
                step = sess.run(step_g)
                saver.save(sess, './tmp/checkpoint/model.ckpt', global_step=step)
    else:
        saver = tf.train.Saver()
        ckpt_path = tf.train.latest_checkpoint('./tmp/checkpoint/')
        saver.restore(sess, ckpt_path)
        for data in data_utils.get_batch(batchsize, training_set[0], training_set[1], training_set[2]):
            _, l, err, p = sess.run([opt, loss, error, pred], {
                x: data[0],
                target: data[1],
                sequence_length: data[2]
            })
            print("loss:{:>10.4f}, edit_distance:{:>8.2f}".format(l, err))
            # convert prediction result from sparse tensor to dense tensor
            dense_pred = sess.run(tf.sparse_to_dense(p[0], p[2], p[1]))
            dense_target = sess.run(tf.sparse_to_dense(data[1][0], data[1][2], data[1][1]))
            print('original:{}'.format(data_utils.to_sentence(dense_target[0], idx2chr)))
            print('predicted:{}'.format(data_utils.to_sentence(dense_pred[0], idx2chr)))
