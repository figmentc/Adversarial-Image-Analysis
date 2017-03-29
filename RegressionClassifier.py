from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
from numpy import *
from scipy.misc import *
import matplotlib.pyplot as plt
import tensorflow as tf
sess = tf.InteractiveSession()

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


def create_tiled_onehot(pos):
    one_hot_i = zeros(10)
    one_hot_i[pos] = 1
    return tile(one_hot_i, (mnist.test.num_examples, 1))
    
# Get correctly classified instances of label '2' prior to adversarial shift
# Returns a boolean tensor to act as mask
def correct_2_mask():
    label_2 = tf.equal(tf.argmax(y_,1), 2)
    correct_predict_2 = tf.logical_and(label_2, correct_prediction)
    return (sess.run(correct_predict_2, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# Calculate delta needed for adversarial shift
def calculate_delta():
    grad_to_six = tf.gradients(cross_entropy, x)
    one_hot_six = zeros(10)
    one_hot_six[6] = 1
    grad_output = sess.run(grad_to_six, feed_dict={x:mnist.test.images, y_:create_tiled_onehot(6)})
    delta = 1000*grad_output[0]
    return delta

# Get positive '6' classifications post adversarial shift
# Returns a boolean tensor to act as mask
def postshift_6_mask():
    delta = calculate_delta()
    new_images = mnist.test.images - delta
    label_6 = tf.equal(tf.argmax(y,1), 6)
    return (sess.run(label_6, feed_dict={x: new_images, y_: mnist.test.labels}))

    
def build_results():
    delta = reshape(calculate_delta(), (mnist.test.num_examples, 28,28))
    prev_input = reshape(mnist.test.images, (mnist.test.num_examples, 28,28))
    postshift_input = reshape(mnist.test.images, (mnist.test.num_examples, 28,28)) - delta
    unfiltered = dstack((prev_input, delta ,postshift_input))
    

    combined_mask = tf.logical_and(correct_2_mask(), postshift_6_mask())
    filtered = tf.boolean_mask(unfiltered,combined_mask)

    filtered_result = sess.run(filtered, feed_dict={})
    print(filtered_result.shape)
    shaped_filtered_result =  reshape(filtered_result, (filtered_result.shape[0]*28, 28*3))[:10*28]
    # mask = tf.logical_and(correct_2_mask(), postshift_6_mask())
    plt.imshow(shaped_filtered_result, cmap='gray')
    plt.show()
    return 

# We will change only correctly classified '2's

# # Calculate gradient of cost towards the '6' classification over input
# loss_to_six = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
# grad_to_six = tf.gradients(loss_to_six, x)
# one_hot_six = zeros(10)
# one_hot_six[6] = 1
# grad_output = sess.run(grad_to_six, feed_dict={x:mnist.test.images, y_:tile(one_hot_six, (mnist.test.num_examples, 1))})
# 
# # Check accuracy from before and after changing input as a factor of the gradient
# If overall accuracy increase, matches our intuition regarding the descent
# six_prediction = tf.equal(tf.argmax(y,1), 6)
# six_acc = tf.reduce_mean(tf.cast(six_prediction, tf.float32))
# 
# print("--Accuracy before descent towards '6':", 
#         six_acc.eval(feed_dict={x: mnist.test.images, y_: tile(one_hot_six, (mnist.test.num_examples, 1))}))
#         
# new_images = mnist.test.images - 1000*grad_output[0]
# 
# print("--Accuracy after descent towards '6':", 
#         six_acc.eval(feed_dict={x: new_images, y_: tile(one_hot_six, (mnist.test.num_examples, 1))}))



# image1 = resize(mnist.test.images[1], (28,28))
# # image2 = resize(new_images[1], (28,28))


# plt.imshow(image1)
# plt.show()
# plt.imshow(image2)
# plt.show()