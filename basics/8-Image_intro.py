import tensorflow as tf
import matplotlib.image as img_plt
import matplotlib.pyplot as plt
import os
image = img_plt.imread(r'D:\IT\SelfTut\PluralSight-Intro_to_Tensor_Flow\Data\images\berkay-gumustekin-ngqyo2AYYnE-unsplash.jpg')
print(image.shape)

X = tf.Variable(image,name="raw_image")
init = tf.global_variables_initializer()
transpose = tf.transpose(X,perm=[1,0,2])
again_transpose = tf.image.transpose_image(transpose)

with tf.Session() as sess:
    sess.run(init)
    out = sess.run(again_transpose)

    plt.imshow(out)
    plt.show()

