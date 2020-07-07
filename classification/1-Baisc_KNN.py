import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

def display_digit_as_image(digit):
    plt.imshow(digit.reshape([28,28]),cmap='Greys',interpolation='nearest')
    plt.show()

mnist = input_data.read_data_sets("mnist_data/")
train_data,train_labels = mnist.train.next_batch(1000)
test_data,test_labels = mnist.test.next_batch(100)

print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)


training_data = tf.placeholder("float",shape=[None,784],name="input_training")
test_entry = tf.placeholder("float",shape=[784],name="test_input")
distance_tensor = tf.reduce_sum(tf.abs(tf.subtract(training_data,test_entry)),axis=1)
predicted_tensor=tf.nn.top_k(tf.negative(distance_tensor),k=1)


if __name__ == '__main__':
    display_digit_as_image(test_data[0])
    print("Corresponding label ="+str(test_labels[0]))
    with tf.Session() as sess:
        _,indices = sess.run(fetches=predicted_tensor,feed_dict={training_data:train_data,test_entry:test_data[3,:]})
        display_digit_as_image(train_data[indices])
        print("Actual label",str(train_labels[indices]))