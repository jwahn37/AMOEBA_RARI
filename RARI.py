# Lab 5 Logistic Regression Classifier
import tensorflow as tf
import numpy as np

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#tf.set_random_seed(777)  # for reproducibility

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

print("RARI.py")
xy = np.loadtxt('RARI_training.txt', delimiter='\t', dtype=np.float32)
#xy = MinMaxScaler(xy)

x_data = xy[:, 5:-1]
y_data = xy[:, -1:]
print("max : ", np.max(x_data, 0))
print("min : ", np.min(x_data, 0))
print (x_data, y_data)
x_data = MinMaxScaler(x_data)

print("MinMaxScaler", x_data)


xy_test = np.loadtxt('RARI_testing.txt', delimiter='\t', dtype=np.float32)
x_data_test = xy_test[:, 5:-1]
y_data_test = xy_test[:, -1:]
x_data_test = MinMaxScaler(x_data_test)

'''
filename_queue = tf.train.string_input_producer(
#        ['dataset_regression_SEI.txt'], shuffle=False, name='filename_queue')
        ['RARI_datas.txt'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default value, in the case of empy clumn. Also specifies the type of the decoded result
record_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.] ]

xy = tf.decode_csv(value, record_defaults=record_defaults, field_delim='\t')

train_x_batch, train_y_batch= \
        tf.train.batch([xy[5:-1], xy[-1:]], batch_size=2000)

print(train_x_batch)
'''
# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#print(type(W), type(b))
#mid = tf.matmul(X,W)
virtual_RARI = tf.matmul(X,W)+b
virtual_RARI_max = np.max(virtual_RARI, 0)

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(-tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        #print("W : ",w, "B : ",bb);
        if step % 10 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a, w, bb= sess.run([hypothesis, predicted, accuracy, W, b],feed_dict={X: x_data, Y: y_data})
    print("\n training : Hypothesis: ", h, "\nCorrect (Y): ", c, "\nY: ",y_data, "\nAccuracy: ", a)
    print("W : ",w, "B : ",bb);

    v_r, v_rm = sess.run([virtual_RARI, virtual_RARI_max], feed_dict={X:x_data, Y:y_data})
    print("virtual RARI : ",v_r, "virtual RARI MAX : ",np.max(v_r, 0), "virtual RARI MIN : ",np.min(v_r, 0))

    print("MODEL: ",w[0][0],w[1][0],w[2][0],bb[0]);
'''
#test with 30%
X_test = tf.placeholder(tf.float32, shape=[None, 3])
Y_test = tf.placeholder(tf.float32, shape=[None, 1])
hypothesis_test = tf.sigmoid(tf.matmul(X_test, w) + bb)


#predicted_test = tf.cast(hypothesis_test > 0.5, dtype=tf.float32)
#accuracy_test = tf.reduce_mean(tf.cast(tf.equal(predicted_test, Y), dtype=tf.float32))

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    t_h= sess.run([hypothesis_test], feed_dict={X:x_data_test, Y:y_data_test})
#    print("\n testing : Hypothesis: ", t_h, "\nCorrect (Y): ", t_p, "\nY: ",y_data_test , "\nAccuracy: ", a)
'''
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
