import tensorflow as tf
import data_helper
from data_helper import next_batch
from tensorflow.contrib import learn
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string('data_file_path', './data/rt-polarity.csv', 'Data source')
tf.flags.DEFINE_string('feature_name', 'comment_text', 'The name of feature column')
tf.flags.DEFINE_string('label_name', 'label', 'The name of label column')
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 2000, "Number of training epochs (default: 200)")

FLAGS = tf.flags.FLAGS


def pre_process():
    # load data
    x_text, y = data_helper.load_data_and_labels(FLAGS.data_file_path, FLAGS.feature_name, FLAGS.label_name)
    # Build vocabulary and cut or extend sentence to fixed length
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    # replace the word using the index of word in vocabulary
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # random shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev


def text_cnn(input_x, input_y, num_classes, dropout_keep_prob, sequence_length, vocab_size, filter_sizes, embedding_size, num_filters):
    # Keeping track of l2 regularization loss (optional)
    l2_loss = tf.constant(0.0)
    # embedding layer
    with tf.name_scope('embedding'):
        embedding = tf.Variable(tf.random_uniform([vocab_size, FLAGS.embedding_dim], -1.0, 1.0), name='word_embedding')
        embedding_chars = tf.nn.embedding_lookup(embedding, input_x)
        embedding_chars_extend = tf.expand_dims(embedding_chars, -1)

    # Create a convolution + maxpool layer for each filter size
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope('conv_maxpool'):
            # convolution layer
            filter_shape = [int(filter_size), embedding_size, 1, num_filters]    # [height, width, in_channel, out_channel]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
            conv = tf.nn.conv2d(embedding_chars_extend, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
            # max pooling
            pooled = tf.nn.max_pool(h, ksize=[1, sequence_length-int(filter_size)+1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool')
            pooled_outputs.append(pooled)
    # combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    # add dropout
    with tf.name_scope('dropout'):
        h_drop = tf.nn.dropout(h_pool_flat, rate=1-dropout_keep_prob)

    # out
    with tf.name_scope('output'):
        W_out = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name='W_out')
        b_out = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_out")
        l2_loss += tf.nn.l2_loss(W) + tf.nn.l2_loss(W_out)
        l2_loss += tf.nn.l2_loss(b) + tf.nn.l2_loss(b_out)
        scores = tf.nn.xw_plus_b(h_drop, W_out, b_out, name='score')
        predictions = tf.argmax(scores, 1, name='predictions')
    return scores, predictions


if __name__ == '__main__':
    x_train, y_train, vocab_processor, x_dev, y_dev = pre_process()
    # train step
    x = tf.placeholder(tf.int32, [None, x_train.shape[1]])
    y_ = tf.placeholder(tf.float32, [None, y_train.shape[1]])
    scores, predictions = text_cnn(x, y_, y_train.shape[1], FLAGS.dropout_keep_prob, x_train.shape[1],
                                   len(vocab_processor.vocabulary_), list(map(int, FLAGS.filter_sizes.split(","))),
                                   FLAGS.embedding_dim, FLAGS.num_filters)
    # loss function
    cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=scores))
    # optimizer
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    # accuracy
    correct_prediction = tf.equal(predictions, tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session() as sess:
        # init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        # train
        for i in range(FLAGS.num_epochs):
            x_batch, y_batch = next_batch(FLAGS.batch_size, x_train, y_train)
            _, acc, loss = sess.run([optimizer, accuracy, cross_entropy], feed_dict={x: x_batch, y_: y_batch})
            if i % 10 == 0:
                print('step {}:, loss: {}, accuracy: {}'.format(i, loss, acc))

        # valid step
        print('valid accuracy: {}'.format(sess.run(accuracy, feed_dict={x: x_dev, y_: y_dev})))

