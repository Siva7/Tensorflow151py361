import tensorflow as tf

adder = tf.Variable(1, dtype=tf.int32, name="Multiplier")
x = tf.Variable(2,dtype=tf.int32,name="Value")
adder.assign_add(100)
y = x.assign(tf.add(x, adder))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        sess.run(y)
    result = sess.run(y)
    with tf.summary.FileWriter('output',sess.graph) as writer:
        pass
    print(result)
