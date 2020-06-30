import tensorflow as tf

X = tf.constant(1)
Y = tf.constant(2)

Z = X+Y
default_graph = tf.get_default_graph()
print(X.graph == default_graph)
print(Y.graph == default_graph)
print(Z.graph == default_graph)

new_graph = tf.Graph()
with new_graph.as_default():
    T = tf.constant(2,name="NewConstant")
    K = Z+Z
    print(K.graph == new_graph)
    print(K.graph == default_graph)
print(default_graph.get_operations())
print(new_graph.get_operations())

# Console Output

"""
True
True
True
False
True
[<tf.Operation 'Const' type=Const>, <tf.Operation 'Const_1' type=Const>, <tf.Operation 'add' type=Add>, <tf.Operation 'add_1' type=Add>]
[<tf.Operation 'NewConstant' type=Const>]
"""