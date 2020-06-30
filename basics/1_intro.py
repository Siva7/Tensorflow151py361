import tensorflow as tf

a = tf.constant(6,name="constant_a",dtype='float32')
b = tf.constant(12,name="constant_b",dtype='float32')
c = tf.constant(24,name="constant_c",dtype='float32')

mul = tf.multiply(a,b,name="mul")
div = tf.divide(c,b,name="div")

addn = tf.add_n([mul,div],name="addn")

square = tf.square(addn,name="squared")

pow = tf.pow(div,3,name="cube")
sqrt = tf.sqrt(pow,name="sqrt"
               )
ses = tf.Session()
writer = tf.summary.FileWriter('output',graph =ses.graph)
res=ses.run(sqrt)
print(addn)
print(res)


ses.close()
writer.close()
