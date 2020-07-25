import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import logging
import numpy as np
logging.basicConfig(level="INFO")


input = tf.placeholder(dtype='float32',shape=[None,784],name="input")
input_labels = tf.placeholder(dtype="int32",shape=[None],name="training_labels")

print("input shape =>"+str(input.shape))
input_reshaped = tf.reshape(input,shape=[-1,28,28,1],name="reshaped_input")
print("input_reshaped =>"+str(input_reshaped.shape))
first_2d_conv = tf.layers.conv2d(input_reshaped,filters=24,kernel_size=3,strides=1,padding="SAME",name="First_convolution")
print("first_2d_conv =>"+ str(first_2d_conv.shape))
first_pooling_layer = tf.layers.max_pooling2d(first_2d_conv,pool_size=3,strides=3,padding='VALID')
print("first_pooling_layer =>"+str(first_pooling_layer.shape))
second_conv_layer  = tf.layers.conv2d(first_pooling_layer,filters=64,kernel_size=4,strides=1,padding="SAME")
print("second_conv_layer =>"+str(second_conv_layer.shape))
logging.info("second_conv_layer =>"+str(second_conv_layer.shape))
flatten_layer = tf.layers.flatten(second_conv_layer)
print("Flatten layer =>"+str(flatten_layer.shape))
output_logit_layer = tf.layers.dense(flatten_layer,10)
print("output_shape =>"+ str(output_logit_layer.shape))
entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_logit_layer,labels=input_labels)
logging.info("entropy shape =>"+str(entropy.shape))
print("entropy shape =>"+str(entropy.shape))
loss = tf.reduce_mean(entropy)

optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)
print("train type =>"+str(type(train)))

prediction = tf.arg_max(output_logit_layer,1)

pick_top = tf.nn.top_k(output_logit_layer,k=1)
print("Pick_top_type=>"+str(type(pick_top)))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    mnist = input_data.read_data_sets("mnist_data")
    sess.run(init)
    for i in range(100):
        train_data,train_label = mnist.train.next_batch(1000)
        _,logit=sess.run(fetches=[train,output_logit_layer],feed_dict={input:train_data,input_labels:train_label})
    for x in range(10):
        print(np.argmax(logit[x]),end="=>")
        print(train_label[x])
