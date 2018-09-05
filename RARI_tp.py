# Lab 5 Logistic Regression Classifier
import tensorflow as tf
import numpy as np

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#tf.set_random_seed(777)  # for reproducibility


#data를 받아와서 parsing한다. input : x_data, output : y_data


xy = np.loadtxt('temp.txt', delimiter='\t', dtype=np.float32)
#xy = np.loadtxt('dataset_regression_SEI.txt', delimiter='\t', dtype=np.float32)

#x_data = xy[:, [0]]
#x_data = xy[:, 5:7]
x_data = np.concatenate((xy[:, [0]], xy[:, 5:7]), axis=1)
y_data = xy[:, [-1]]

'''
filename_queue = tf.train.string_input_producer(
        ['dataset_regression_SEI.txt'], shuffle=False, name='filename_queue')
'''
filename_queue = tf.train.string_input_producer(
        ['data-01-test-score.csv'], shuffle=False, name='filename_queue')
'''
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default value, in the case of empy clumn. Also specifies the type of the decoded result
# record_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.] ]
record_defaults = [[0.], [0.], [0.], [0.] ]
xy = tf.decode_csv(value, record_defaults=record_defaults)


train_x_batch, train_y_batch= \
        tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)
print("hello")
#print(type(train_x_batch))
#print(x_data)
#print(y_data)
#print(x_data.shape, y_data.shape)
'''
# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(-tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
# Start populating the filename queue.

#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(2001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})

#        x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
#        cost_val, _ = sess.run([cost, train], feed_dict={X: x_batch, Y: y_batch})

        if step % 1000 == 0:
            print(step, cost_val)

#    coord.request_stop()
#    coord.join(threads)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       #feed_dict={X: x_batch, Y: y_batch})
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

'''
0 0.82794
200 0.755181
400 0.726355
600 0.705179
800 0.686631
...
9600 0.492056
9800 0.491396
10000 0.490767
...
 [ 1.]
 [ 1.]
 [ 1.]]
Accuracy:  0.762846
'''
