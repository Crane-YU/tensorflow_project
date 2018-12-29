import tensorflow as tf
#
# w1 = tf.Variable(0, dtype=tf.float32)
# num_updates = tf.Variable(0, trainable=False)
#
# ema = tf.train.ExponentialMovingAverage(0.99)
# maintain_average_op = ema.apply([w1])
#
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#
#     # Initialize the value for w1 and the ema of w1 to be 0
#     sess.run(tf.assign(w1, 10))
#     sess.run(maintain_average_op)
#     print(sess.run([w1, ema.average(w1)]))
#
#     # Change the value of w1 to 5
#     sess.run(tf.assign(w1, 5))
#     # Update the ema value of w1
#     sess.run(maintain_average_op)
#     print(sess.run([w1, ema.average(w1)]))
#
#     # Update the value of num_updates to be 10000
#     sess.run(tf.assign(num_updates, num_updates))
#     # Change the value of w1 to 10
#     sess.run(tf.assign(w1, 10))
#     # Update the ema value of w1
#     sess.run(maintain_average_op)
#     print(sess.run([w1, ema.average(w1)]))
#
w = tf.Variable(1.0)
ema = tf.train.ExponentialMovingAverage(0.9)
update = tf.assign_add(w, 1.0)

with tf.control_dependencies([update]):
    # 返回一个op, 这个op用来更新moving_average, i.e. shadow value
    ema_op = ema.apply([w])  # 这句和下面那句不能调换顺序

# 以 w 当作 key, 获取 shadow value 的值
ema_val = ema.average(w)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(3):
        sess.run(ema_op)
        print(sess.run(ema_val))
