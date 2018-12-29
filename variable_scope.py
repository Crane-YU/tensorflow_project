import tensorflow as tf

with tf.variable_scope("root"):
    print(tf.get_variable_scope().reuse)

    with tf.variable_scope("foo", reuse=True):
        print(tf.get_variable_scope().reuse)

        with tf.variable_scope("bar"):
            print(tf.get_variable_scope().reuse)

    print(tf.get_variable_scope().reuse)

v1 = tf.get_variable("v", [1])
print(v1.name)

with tf.variable_scope("foo"):
    v2 = tf.get_variable("v", [1])
    print(v2.name)

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3 = tf.get_variable("v", [1])
        print(v3.name)
