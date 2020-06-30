import tensorflow as tf

x = tf.constant([100,200,300],name="x")
y = tf.constant([1,2,3],name="y")

summ = tf.reduce_sum(x,name="reduce_sum")
prod = tf.reduce_prod(y,name="reduce_prod")

avg = tf.add_n([summ,prod],name="add_scalars")
div = tf.div(summ,prod,name = "div")

final = tf.subtract(avg,div,name="final")


sess = tf.Session()
res= sess.run(final)
print(res)
writer = tf.summary.FileWriter(logdir='output',graph=sess.graph)
writer.flush()
writer.close()
sess.close()
