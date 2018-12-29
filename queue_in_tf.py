import tensorflow as tf

# Create a queue with max 2 elements in the queue
q = tf.FIFOQueue(2, "int32")

# Initialize the queue
init = q.enqueue_many(([0, 10],))

x = q.dequeue()
y = x + 1

q_inc = q.enqueue([y])

with tf.Session() as sess:
    init.run()
    for times in range(5):
        output_val, input_op = sess.run([x, q_inc])
        print(output_val)
