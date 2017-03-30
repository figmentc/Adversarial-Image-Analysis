from tensorflow.examples.tutorials.mnist import input_data
from numpy import *
from scipy.misc import *
import matplotlib.pyplot as plt
import tensorflow as tf
### Google Tensorflow ConvNet

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

sess = tf.InteractiveSession()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
                        
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
sess.run(tf.global_variables_initializer())

y=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(3000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    print("val accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0}))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})
print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        
### Adversarial Image Generator

# Returns the tiled onehot ouput of classification pos from 0-9
# int -> numpy array
def create_tiled_onehot(pos):
    one_hot_i = zeros(10)
    one_hot_i[pos] = 1
    return tile(one_hot_i, (mnist.test.num_examples, 1))

# Return the "shift" needed to generate the adversarial image as a factor of alpha 
# and the gradient
# int -> float32 tensor
def calculate_delta(alpha):
    
    #calculate gradient of the cost over input
    grad_to_six = tf.gradients(cross_entropy, x)
    grad_output = sess.run(grad_to_six, feed_dict={x:mnist.test.images, y_:create_tiled_onehot(6), keep_prob: 1.0})
    
    # delta calculated as gradient of the cost over input scaled by factor alpha
    delta = alpha*grad_output[0]
    return delta
    
# Return boolean mask for correctly classified instances of lablel '2' within the mnist test set
# -> boolean tensor
def correct_2_mask():
    label_2 = tf.equal(tf.argmax(y_,1), 2)
    correct_predict_2 = tf.logical_and(label_2, correct_prediction)
    return (sess.run(correct_predict_2, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


# Return boolean mask for classified instances of any label under under "6" after the adversarial shift
# int -> boolean tensor
def postshift_6_mask(alpha):
    delta = calculate_delta(alpha)
    new_images = mnist.test.images - delta
    label_6 = tf.equal(tf.argmax(y,1), 6)
    return (sess.run(label_6, feed_dict={x: new_images, y_: mnist.test.labels, keep_prob: 1.0}))

# Output the first 10 input under label '2' classified as '6' due to the adversarial shift
# Returns the total number of input affected by adversarial shift (2 -> 6)
# int -> float32 tensor
def build_results(alpha):
    # Concatenate all original input, delta, and postshift_input
    delta = reshape(calculate_delta(alpha), (mnist.test.num_examples, 28,28))
    prev_input = reshape(mnist.test.images, (mnist.test.num_examples, 28,28))
    postshift_input = reshape(mnist.test.images, (mnist.test.num_examples, 28,28)) - delta
    unfiltered = dstack((prev_input, delta ,postshift_input))

    # Generate mask and filter for those affected
    combined_mask = tf.logical_and(correct_2_mask(), postshift_6_mask(alpha))
    filtered = tf.boolean_mask(unfiltered,combined_mask)
    filtered_result = sess.run(filtered, feed_dict={})

    # Reshape filtered_result to generate image from pixels
    shaped_filtered_result =  reshape(filtered_result, (filtered_result.shape[0]*28, 28*3))
    
    return shaped_filtered_result

def generate_graph():
    alphas = [200, 300, 400, 700, 1000, 1500]
    f, axarr = plt.subplots(ncols=len(alphas))
    for i in range(len(alphas)):
        axarr[i].imshow(build_results(alphas[i])[:10*28], cmap="gray")
        axarr[i].axis("off")
        axarr[i].set_title(alphas[i])
    plt.show()
    
