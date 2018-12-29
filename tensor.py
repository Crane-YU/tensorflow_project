import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant([1., 2.], name='a')
b = tf.constant([2., 3.], name='b')

result = tf.add(a, b, name='add')
print(result)

# 1st way to get the value from a tensor
with tf.Session() as sess:
    print(sess.run(result))
    print(result.eval(session=sess))

# 2nd way to get the value from a tensor
sess = tf.Session()
with sess.as_default():
    print(result.eval())
sess.close()

# 3rd way to get the value
sess = tf.InteractiveSession()
print(result.eval())
sess.close()
