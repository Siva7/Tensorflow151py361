import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level="INFO")

mnist=input_data.read_data_sets("mnist_data")


input = tf.placeholder(dtype=tf.float32,shape=[None,784],name="input")
output_labels = tf.placeholder(dtype=tf.int32,shape=[None],name="labels")
reshaped_input = tf.reshape(input,shape=[-1,28,28])
basic_cell = tf.contrib.rnn.BasicRNNCell(150)

output,states = tf.nn.dynamic_rnn(basic_cell,inputs=reshaped_input,dtype=tf.float32)

logits = tf.layers.dense(states,10)

predection = tf.arg_max(logits,1)

accuracy = tf.metrics.accuracy(predictions=predection,labels=output_labels)

entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=output_labels)

loss = tf.reduce_mean(entropy)

optimizer = tf.train.AdamOptimizer()

training = optimizer.minimize(loss)

init = tf.global_variables_initializer()
init_loc = tf.local_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess.run(init_loc)

    test_input, test_label = mnist.test.next_batch(100)
    for j in range(20):
        train_input, train_label = mnist.train.next_batch(100)
        for i in range(100):
            _,acc = sess.run(fetches=[training,accuracy],feed_dict={input:train_input,output_labels:train_label})
            logging.info("Accuracy at iter "+str(i) +"/"+str(j)+" = "+str(acc))

    acc = sess.run(fetches=[accuracy], feed_dict={input: test_input, output_labels: test_label})
    logging.info("Test Accuracy =>" + str(acc))