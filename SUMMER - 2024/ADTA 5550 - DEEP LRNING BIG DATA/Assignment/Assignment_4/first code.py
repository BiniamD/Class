
# BINIAM ABEBE


# CNN for image Recognition with MNIST Dataset


import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
%matplotlib inline


import tensorflow as tf


#version check
print(tf.__version__)


#load minst dataset from tensorflow example
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True);



type(mnist)


mnist.train.images.shape
mnist.test.images.shape


# Initialize weights in Filter



def initialize_weights(filter_shape):
    """
    Initializes the weights of a neural network filter.

    Args:
        filter_shape (tuple): The shape of the filter weights.

    Returns:
        tf.Variable: The initialized filter weights.

    """
    init_random_dist = tf.truncated_normal(filter_shape, stddev=0.1)
    
    return tf.Variable(init_random_dist)


# Initialize bias


# def initalize_bias
def initialize_bias(bias_shape):
    """
    Initializes the bias of a neural network layer.

    Args:
        bias_shape (tuple): The shape of the bias.

    Returns:
        tf.Variable: The initialized bias.

    """
    initial_bias_vals = tf.constant(0.1, shape=bias_shape)
    
    return tf.Variable(initial_bias_vals)


# set up Convlutional Layer and Perform Convolution Computation


# def create_conv_layer
def create_conv_layer(input_data, filter_shape):
    """
    Creates a convolutional layer in a neural network.

    Args:
        input_data (tf.Tensor): The input to the layer.
        filter_shape (tuple): The shape of the filter weights.

    Returns:
        tf.Tensor: The output of the layer.

    """
    filter_weights = initialize_weights(filter_shape)

    #create a convolutional layer
    Conv_lyaer_output = tf.nn.conv2d(input=input_data, filter=filter_weights, strides=[1, 1, 1, 1], padding='SAME')

    return Conv_lyaer_output


# set up a Relu Layer and Perform Computation : Dot Product + Bias (x.w + b)


# create relu_layer and compute dot product

def create_relu_layer_and_compute_dot_product(input_data, filter_shape):
    """
    Creates a ReLU layer in a neural network and computes the dot product.

    Args:
        input_data (tf.Tensor): The input to the layer.
        filter_shape (tuple): The shape of the filter weights.

    Returns:
        tf.Tensor: The output of the layer.

    """
    bias = initialize_bias([filter_shape[3]])
    
    relu_layer_output = tf.nn.relu(input_data + bias)

    return relu_layer_output


# Set up a Pooling Layer and reduce Spatial size


# create max_pooling_layer
def create_max_pooling_layer(input_data):
    """
    Creates a max pooling layer in a neural network.

    Args:
        input_data (tf.Tensor): The input to the layer.

    Returns:
        tf.Tensor: The output of the layer.

    """
    return tf.nn.max_pool(input_data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Setup Fully Connected Layer and Perform Computatin: (input * weights) + Bias


# create fully_connected_layer
def create_fully_connected_layer(input_data, output_size):
    """
    Creates a fully connected layer in a neural network.

    Args:
        input_data (tf.Tensor): The input to the layer.
        output_size (int): The size of the output.

    Returns:
        tf.Tensor: The output of the layer.

    """
    input_size = int(input_data.get_shape()[1])
    
    weights = initialize_weights([input_size, output_size])
    bias = initialize_bias([output_size])
    
    return tf.matmul(input_data, weights) + bias


# Phase I : Build the Convolution Neural Network


# Create placeholders for inputs and Labels : x & y_true


#place holder for input data
x = tf.placeholder(tf.float32, shape=[None, 784])


# place holder for y
y_true = tf.placeholder(tf.float32, shape=[None, 10])


# Reshape the Input Placeholder X


#reshape the input data
x_image = tf.reshape(x, [-1, 28, 28, 1])


# Create 1st Convlutional Layer,ReLu Layer and perform Computation : x*W +b


# create first convolutional layer
# 5x5 convolutional layer with 32 filters

conv_layer_1 = create_conv_layer(x_image, filter_shape=[5, 5, 1, 32])

# create first relu layer
relu_layer_1 = create_relu_layer_and_compute_dot_product(conv_layer_1, filter_shape=[5, 5, 1, 32])


# Create 1st Pooling Layer and Reduce Spatial size


# create first max pooling layer
max_pooling_layer_1 = create_max_pooling_layer(relu_layer_1)


# Create 2nd Convlutional Layer,ReLu Layer and perform Computation : x*W +b


# create second convolutional layer
# 5x5 convolutional layer with 64 filters
conv_layer_2 = create_conv_layer(max_pooling_layer_1, filter_shape=[5, 5, 32, 64])

# create second relu layer
relu_layer_2 = create_relu_layer_and_compute_dot_product(conv_layer_2, filter_shape=[5, 5, 32, 64])


# Create 2nd Pooling Layer and Reduce Spatial size


# create second max pooling layer
max_pooling_layer_2 = create_max_pooling_layer(relu_layer_2)


# Reshape/Flatten Data Making it Ready to be Fed into 1st fc Layer


# reshape the max pooling layer
max_pooling_layer_2_flat = tf.reshape(max_pooling_layer_2, [-1, 7*7*64]) # 7*7*64 = 3136


# Create 1st FC Layer, Relu Layer , and output Data to Dropout Layer


# create first fully connected layer

fully_connected_layer_1 = create_fully_connected_layer(max_pooling_layer_2_flat, output_size=1024) 

# create first relu layer output
fully_connected_layer_1_relu = tf.nn.relu(fully_connected_layer_1)


# Create Dropput Layer and Dropput a fraction of output Randomly


# create dropout layer
keep_prob = tf.placeholder(tf.float32)

# dropout layer
dropout_layer = tf.nn.dropout(fully_connected_layer_1_relu, keep_prob=keep_prob)


# Create Final FC Layer, Compute (x.W + B), and Produce Finale Ouptus


# create second fully connected layer as pred
y_pred = create_fully_connected_layer(dropout_layer, output_size=10)


# Define Loss Function and Calaculate softmax Cross Entropy Loss


# create cross entropy loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))


# Create an optimizer


#get optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)


# Create a Trainer to Traing CNN Model


#create train step
train = optimizer.minimize(cross_entropy)


# Train and Test CNN Deep Learning Model on MNIST Dataset


# Initalize all Variables


#initialize variables
init = tf.global_variables_initializer()


# steps
steps = 5000


# Run tf.sessioon () to Train and Test Deep Learning CNN Model


# create session and run the model 
with tf.Session() as sess:

    # initialize the session
    sess.run(init)
    
    for i in range(steps):
        # get the next batch of data from mnist dataset 
        batch_x, batch_y = mnist.train.next_batch(50)
        
        # run the train step
        sess.run(train, feed_dict={x: batch_x, y_true: batch_y, keep_prob: 0.5})

        # print out a message every 100 steps

        if i % 100 == 0:
            print('Currently on step {}'.format(i))
            print('Accuracy is:')

            # Test the Train Model
            matchs = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

            # convert matchs to float32 and calculate mean
            # to get the accuracy
            accuracy = tf.reduce_mean(tf.cast(matchs, tf.float32))

            # test the model at 100 steps
            test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, \
                                                           y_true: mnist.test.labels, \
                                                           keep_prob: 1.0})
            print(test_accuracy)
            print('\n')            



