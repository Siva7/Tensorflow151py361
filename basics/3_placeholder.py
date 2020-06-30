import tensorflow as tf
from tensorflow import int32

x = tf.placeholder(dtype=int32,shape=[4],name="x")
y = tf.placeholder(dtype=int32,shape=[4],name="y")

x_ = tf.reduce_sum(x,name="red_sum")
y_ = tf.reduce_sum(y,name="red_sum")
divs = tf.div(x_,y_,name="div")
divs_sum = tf.reduce_sum(divs,name="div_sum")
sess = tf.Session()
a = tf.placeholder(dtype=int32,shape=[],name="scalar_test")
b = tf.square(a,name="square")
sess.run(divs,feed_dict={x:[2,4,6,8],y:[1,2,3,4]})

[res1,res2]=sess.run(fetches=[divs_sum,b],feed_dict={divs:8,a:4})

print(res1)
print(res2)

writer = tf.summary.FileWriter('output',graph=sess.graph)


writer.close()
sess.close()
