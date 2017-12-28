import tensorflow as tf
import data_utils
import params

batch_size = 64
lr = 0.001
epoch_num = 100
display_step = 10

graph = tf.Graph()
with graph.as_default():
    # get dataset
    d = data_utils.get_files('./data/spoken_numbers_pcm', 1)
    arr_x, arr_y = data_utils.get_dataset(d)
    # training set
    data_x_train = tf.data.Dataset.from_tensor_slices(arr_x[128:])
    data_y_train = tf.data.Dataset.from_tensor_slices(arr_y[128:])
    training_set = tf.data.Dataset.zip((data_x_train, data_y_train)).batch(batch_size).shuffle(512)
    iterator_train = training_set.make_initializable_iterator()
    ne_train = iterator_train.get_next()
    # validation set
    data_x_val = tf.data.Dataset.from_tensor_slices(arr_x[:128])
    data_y_val = tf.data.Dataset.from_tensor_slices(arr_y[:128])
    validation_set = tf.data.Dataset.zip((data_x_val, data_y_val)).batch(128)
    iterator_validation = validation_set.make_initializable_iterator()
    ne_validation = iterator_validation.get_next()

    # define the placeholder
    x = tf.placeholder(tf.float32, [None, 168, 13, 1])
    y = tf.placeholder(tf.float32, [None, 10])

    # conv layer
    with tf.name_scope("conv_layer"):
        conv1 = tf.layers.conv2d(x, 32, 5, strides=1, padding='same', name='conv1', activation=params.act)
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2, name='pool1')
        conv2 = tf.layers.conv2d(pool1, 64, 5, strides=1, padding='same', name='conv2', activation=params.act)
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2, name='pool2')
        conv3 = tf.layers.conv2d(pool2, 128, 5, strides=1, padding='same', name='conv3', activation=params.act)
        pool3 = tf.layers.max_pooling2d(conv3, 2, 2, name='pool3')

    # reshape
    with tf.name_scope("reshape"):
        pool3_shape = tf.shape(pool3)
        pool3_flat = tf.reshape(pool3, [-1, 2688])

    with tf.name_scope("dnn"):
        dense1 = tf.layers.dense(pool3_flat, 128, activation=params.act, name='dense1')
        dense2 = tf.layers.dense(dense1, 64, activation=params.act, name='dense2')

    with tf.name_scope("fc"):
        fc = tf.layers.dense(dense2, 10, name='fc1')

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=fc))
    opt = tf.train.AdamOptimizer(lr).minimize(loss)

    prediction = tf.argmax(tf.nn.softmax(fc), axis=1)
    real_label = tf.argmax(y, axis=1)

    error_rate = 1. - tf.reduce_mean(tf.cast(tf.equal(prediction, real_label), tf.float32))
    tf.summary.scalar('error_rate', error_rate)
    tf.summary.scalar('loss', loss)

with tf.Session(graph=graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(iterator_train.initializer)
    writer_train = tf.summary.FileWriter('./tmp/train', graph=graph)
    writer_vali = tf.summary.FileWriter('./tmp/vali', graph=graph)
    merged = tf.summary.merge_all()
    epoch = 1
    step = 0
    while epoch < epoch_num:
        try:
            # training
            data = sess.run(ne_train)
            audio = data[0]
            mfcc = list(map(data_utils.process, audio))
            _, l, err, mgd = sess.run([opt, loss, error_rate, merged], {x: mfcc, y: data[1]})
            step += 1
            writer_train.add_summary(mgd, global_step=step)

            if step % display_step == 0:
                print("Epoch:{}, Step:{}, Loss:{:.4f}, Error Rate:{:.2%}".format(epoch, step, l, err))

            # validation
            sess.run(iterator_validation.initializer)
            data = sess.run(ne_validation)
            audio = data[0]
            mfcc = list(map(data_utils.process, audio))
            l, err, mgd = sess.run([loss, error_rate, merged], {x: mfcc, y: data[1]})
            writer_vali.add_summary(mgd, global_step=step)



        except tf.errors.OutOfRangeError:
            sess.run(iterator_train.initializer)
            epoch += 1
            continue
