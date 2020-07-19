import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
def get_manhattan_distance_tensor(tensor_n,tensor_1):
    return tf.reduce_sum(tf.abs(tf.subtract(tensor_n, tensor_1)),axis=1)
def get_euclidean_distance_tensor(tensor_n, tensor_1):
    return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tensor_n, tensor_1)),axis=1))

mnist = input_data.read_data_sets("mnist_data/")
train_data,train_labels = mnist.train.next_batch(1000)
test_data,test_labels = mnist.test.next_batch(100)

test_input = tf.placeholder(dtype="float",shape=[784],name="Test_input")
train_input = tf.placeholder(dtype="float",shape=[None,784],name="Train_input")

accuracy = 0
for i in range(0,100):
    distance = get_manhattan_distance_tensor(train_input,test_input)
    top_n = tf.nn.top_k(tf.negative(distance),k=1)
    with tf.Session() as sess:
        _,index = sess.run(fetches=top_n,feed_dict={test_input:test_data[i,:],train_input:train_data})
    if train_labels[index] == test_labels[i]:
        accuracy = accuracy+(1/100)
    print("Accuracy at index "+str(i)+" is "+str(accuracy))

accuracy = 0
for i in range(0,100):
    distance = get_euclidean_distance_tensor(train_input,test_input)
    top_n = tf.nn.top_k(tf.negative(distance),k=1)
    with tf.Session() as sess:
        _,index = sess.run(fetches=top_n,feed_dict={test_input:test_data[i,:],train_input:train_data})


    if train_labels[index] == test_labels[i]:
        accuracy = accuracy+(1/100)
    print("Accuracy at index "+str(i)+" is "+str(accuracy))

with tf.Session() as sess:
    writer = tf.summary.FileWriter('output', sess.graph)
    writer.close()
