import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

input_node = 784
output_node = 10

layer1_node = 500
batch_size = 100
learning_rate_base = 0.8
learning_rate_decay = 0.99

regularization_rate = 0.0001
training_steps = 30000
moving_average_decay = 0.99


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2

    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + biases1)
        return tf.matmul(layer1, avg_class.average(weights2)) + biases2


def train(mnist):
    x = tf.placeholder(tf.float32, [None, input_node], name="x-input")
    y = tf.placeholder(tf.float32, [None, output_node], name="y-input")

    # Generate the hidden layer parameters
    weights1 = tf.Variable(tf.truncated_normal([input_node, layer1_node], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[layer1_node]))
    weights2 = tf.Variable(tf.truncated_normal([layer1_node, output_node], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[output_node]))

    y_hat = inference(x, None, weights1, biases1, weights2, biases2)

    # Define the variable to store the training times
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    # tf.trainable_variables() returns a list of Variable objects
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    average_y_hat = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # Calculate the cross entropy
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat, labels=tf.argmax(y, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # Calculate the L2 regularization function
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)

    # Only calculate the regularization loss for weights
    regularization = regularizer(weights1) + regularizer(weights2)

    # Calculate the total loss
    loss = cross_entropy_mean + regularization

    learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, mnist.train.num_examples/batch_size,
                    learning_rate_decay)

    # Optimize the loss function
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    # Update the weights and biases
    train_op = tf.group(train_step, variable_averages_op)

    # Calculate the accuracy of prediction
    correct_prediction = tf.equal(tf.argmax(average_y_hat, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize session and start training
    with tf.Session() as sess:
        tf.global_variables_initializer.run()

        validate_feed = {x: mnist.validation.images, y: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y: mnist.test.labels}

        for i in range(training_steps):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training steps, validation accuracy using average model is %g" % (i, validate_acc))

            # Generate the batch for this turn and train
            xs, ys = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={x: xs, y: ys})

        # Test the model on the testing data set
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training steps, test accuracy using average model is %g" % (training_steps, test_acc))


def main(argv=None):
    mnist = input_data.read_data_sets("/Users/craneyu/Desktop/tensorflow", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
