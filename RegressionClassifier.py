from tensorflow.examples.tutorials.mnist import input_data
from numpy import *
from scipy.misc import *
import matplotlib.pyplot as plt
import tensorflow as tf

sess = tf.InteractiveSession()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

### Google Tensorflow Sample Code 

# Input Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Weights 
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer())

# Output
y = tf.matmul(x,W) + b

# Loss Function
softmax = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
cross_entropy = tf.reduce_mean(softmax)

#Accuracy Calc
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Training
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for i in range(5000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  
  # Uncomment to get Training Set Accuracy every 500 iters
  
  # if i%500==0:
  #   print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



### Adversarial Image Analysis

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
    grad_output = sess.run(grad_to_six, feed_dict={x:mnist.test.images, y_:create_tiled_onehot(6)})
    
    # delta calculated as gradient of the cost over input scaled by factor alpha
    delta = alpha*grad_output[0]
    return delta
    
# Return boolean mask for correctly classified instances of lablel '2' within the mnist test set
# -> boolean tensor
def correct_2_mask():
    label_2 = tf.equal(tf.argmax(y_,1), 2)
    correct_predict_2 = tf.logical_and(label_2, correct_prediction)
    return (sess.run(correct_predict_2, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# Return boolean mask for classified instances of any label under under "6" after the adversarial shift
# int -> boolean tensor
def postshift_6_mask(alpha):
    delta = calculate_delta(alpha)
    new_images = mnist.test.images - delta
    label_6 = tf.equal(tf.argmax(y,1), 6)
    return (sess.run(label_6, feed_dict={x: new_images, y_: mnist.test.labels}))

# Output the first 10 input under label '2' classified as '6' due to the adversarial shift
# Returns the total number of input affected by adversarial shift (2 -> 6)
# int -> int
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
    shaped_filtered_result =  reshape(filtered_result, (filtered_result.shape[0]*28, 28*3))[:10*28]
    
    plt.imshow(shaped_filtered_result, cmap='gray')
    plt.show()
    
    return filtered_result.shape[0]
    
if __name__ == __main__:
    print(build_results(results(500)))