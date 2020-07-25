import tarfile
import os
import logging
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
logging.basicConfig(level="INFO")

if not os.path.isdir('cifar-10-batches-py'):
    logging.info("Extracting as data is not present")
    with tarfile.open('cifar/cifar-10-python.tar.gz') as tar:
        tar.extractall()
        tar.close()

# Get label map from metadata and also convert all the labels to strings from bytes
def get_label_name_mapping():
    label_map={}
    with open('cifar-10-batches-py/batches.meta',mode='rb') as file:
        label_dict = pickle.load(file)
        for label in label_dict:
            label_map[str(label)] = label_dict[label]
    return label_map['label_names']
label_name = get_label_name_mapping()

logging.info(label_name)


def load_cifar_data_of_batch(batch_id):
    file_name = 'cifar-10-batches-py/data_batch_'+str(batch_id)
    if batch_id == "test":
        file_name = "cifar-10-batches-py/test_batch"
    with open(file_name,mode='rb') as file:
        batch = pickle.load(file,encoding='bytes')
        features = batch[b'data'].reshape(len(batch[b'data']),3,32,32).transpose(0,2,3,1)
        labels = batch[b'labels']
        return features,labels

def plot_an_image_at_random_and_print_its_label(features,labels,actual=None):
    rand_index = np.random.randint(0, len(features))
    plt.imshow(features[rand_index])
    if actual:
        title=str("Pred ="+label_name[labels[rand_index]] + ";Actual ="+label_name[actual[rand_index]])
    plt.title(title)
    plt.show()


# batch_one_feature , batch_one_label = load_cifar_data_of_batch("1")
# plot_an_image_at_random_and_print_its_label(batch_one_feature,batch_one_label)

input = tf.placeholder(dtype=tf.float32,shape=[None,32,32,3],name="input")
output_labels = tf.placeholder(dtype=tf.int32,shape=[None],name="Actual_Labels")

first_2dconv_layer = tf.layers.conv2d(input,filters=32,kernel_size=3,strides=3,padding='SAME',name="first_2d_conv_layer")
logging.info(first_2dconv_layer.shape)
second_2dconv_layer = tf.layers.conv2d(first_2dconv_layer,filters=64,kernel_size=3,strides=3,padding='SAME')
first_pooling_layer = tf.layers.max_pooling2d(second_2dconv_layer,pool_size=2,strides=2,padding='SAME',name="pooling_layer")
logging.info(first_pooling_layer.shape)
flatten_layer = tf.layers.flatten(first_pooling_layer,name="Flatten_layer")
logging.info("Flatten_layer_shape ="+str(flatten_layer.shape))
logit_layer = tf.layers.dense(flatten_layer,10,name="logit_layer")

entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit_layer,labels=output_labels)
loss = tf.reduce_mean(entropy)
optimizer = tf.train.AdamOptimizer()
training = optimizer.minimize(loss)

predection = tf.argmax(logit_layer,1)
logging.info("Logit_layer_shape = "+str(logit_layer.shape))
logging.info("Precetion shape ="+str(predection.shape))
accuracy = tf.metrics.accuracy(labels=output_labels,predictions=predection)
init = tf.global_variables_initializer()
init_l  = tf.local_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for i in range(50):
        for batch in range(1,6):
            sess.run(init_l)
            logging.info("Training with Batch "+str(batch)+"/"+str(i))
            features,labels = load_cifar_data_of_batch(batch)
            train_res ,accc  = sess.run(fetches=[training,accuracy],feed_dict={input:features,output_labels:labels})
            logging.info("Training Accuracy = "+str(accc))
    sess.run(init_l)
    test_features,test_labels = load_cifar_data_of_batch("test")
    pred,acc = sess.run(fetches=[predection,accuracy],feed_dict={input:test_features,output_labels:test_labels})
    logging.info("Testing Accuracy"+str(acc))
    plot_an_image_at_random_and_print_its_label(test_features,pred,test_labels)
    plot_an_image_at_random_and_print_its_label(test_features,pred,test_labels)
    plot_an_image_at_random_and_print_its_label(test_features, pred, test_labels)
    plot_an_image_at_random_and_print_its_label(test_features, pred, test_labels)
    plot_an_image_at_random_and_print_its_label(test_features, pred, test_labels)
    plot_an_image_at_random_and_print_its_label(test_features, pred, test_labels)