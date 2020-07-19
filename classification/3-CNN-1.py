import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def display_digit(data,label):
    plt.title("Label ="+str(label))
    data_reshaped = data.reshape(28,28)
    plt.imshow(data_reshaped,cmap='Greys',interpolation='nearest')


mnist  = input_data.read_data_sets("mnist_data")

print("start")

input_to_cnn = tf.placeholder(dtype=tf.float32,shape=[None,784],name="raw_input_layer")
train_labels =  tf.placeholder(dtype=tf.int32,shape=[None],name="training_labels")

reshape_input = tf.reshape(input_to_cnn,shape=[-1,28,28,1],name="reshaped_input")
first_conv2d =  tf.layers.conv2d(reshape_input,filters=32,kernel_size=3,strides=1,padding='SAME',name="first_conv")
second_conv2d =  tf.layers.conv2d(first_conv2d,filters=64,kernel_size=3,strides=2,padding='SAME',name="second_conv")
first_pooling =  tf.layers.max_pooling2d(second_conv2d,pool_size=3,strides=3,name="pooling")
flatten_layer = tf.reshape(first_pooling,shape=[-1,1024],name="flatten_layer")
logits = tf.layers.dense(flatten_layer,10,name="logits")

top_id = tf.nn.top_k(logits,1)

entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=train_labels)

loss = tf.reduce_mean(entropy)

optimizer =  tf.train.AdamOptimizer()

optimization = optimizer.minimize(loss)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)

    for i in range(100):
        training_data, training_labels = mnist.train.next_batch(1000)
        sess.run(fetches=[optimization],feed_dict={input_to_cnn:training_data,train_labels:training_labels})

    testing_data, testing_labels = mnist.test.next_batch(10)
    result, top_id = sess.run(fetches=[logits, top_id], feed_dict={input_to_cnn: testing_data,train_labels:testing_labels})

    print(result)
    print(top_id)
    value , index = top_id

    print("--------------")
    print(testing_labels)
    print(index)


