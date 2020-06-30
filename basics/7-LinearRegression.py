import tensorflow as tf

W = tf.Variable([0.3],tf.float32,name="W")
X = tf.placeholder(dtype=tf.float32,name="X")
B = tf.Variable([-0.3],tf.float32,name="B")

Y_REAL = tf.placeholder(dtype=tf.float32,name="y_real")
linear_model = W*X + B


loss = tf.reduce_sum(tf.square(tf.subtract(linear_model,Y_REAL)))

optimizer = tf.train.GradientDescentOptimizer(0.01)

optimization = optimizer.minimize(loss)

init = tf.global_variables_initializer()

x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
       w,b,o= sess.run([W,B,optimization],feed_dict={X:x_train,Y_REAL:y_train})
       print("Run_id :"+str(i)+ "  W = "+str(w) +", B="+str(b))

    w,b,o = sess.run(fetches=[W,B,optimization],feed_dict={X:x_train,Y_REAL:y_train})
    print("Final :" + "  W = " + str(w) + ", B=" + str(b))