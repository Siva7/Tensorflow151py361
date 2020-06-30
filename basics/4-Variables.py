import tensorflow as tf

x = tf.placeholder(dtype=tf.float32,name="X")
y = tf.Variable([0,0],dtype=tf.float32,name="y")

W = tf.constant([2.4,3.5],name="W",dtype=tf.float32)
b = tf.constant([4.5,6.5],name="b",dtype=tf.float32)

y = W*x + b

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    y_result = sess.run(y,feed_dict={x:[6.7,8.7]})
    print(y_result)
    with  tf.summary.FileWriter('output',graph=sess.graph) as writer:
            print("Graph Visualization")


S = W * x
init_partial = tf.local_variables_initializer()

with tf.Session() as sess:
    sess.run(init_partial)
    s_result = sess.run(S,feed_dict={x:[10,20]})
    print(s_result)
    with tf.summary.FileWriter('output2',graph=sess.graph) as writer:
        pass


