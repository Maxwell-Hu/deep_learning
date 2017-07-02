import pickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None,real_dim), name = 'inputs_real') # None denote the batch size
    inputs_z = tf.placeholder(tf.float32, (None,z_dim), name = 'inputs_z')

    return inputs_real, inputs_z

def generator(z, out_dim, n_units=128, reuse=False,  alpha=0.01):
    ''' Build the generator network.

        Arguments
        ---------
        z : Input tensor for the generator
        out_dim : Shape of the generator output
        n_units : Number of units in hidden layer
        reuse : Reuse the variables with tf.variable_scope
        alpha : leak parameter for leaky ReLU

        Returns
        -------
        out, logits:
    '''
    with tf.variable_scope('generator',reuse=reuse)# finish this
        # Hidden layer
        h1 = tf.layers.dense(z,n_units,activation=None)  # A fully connected layer
        # Leaky ReLU
        h1 = tf.maximum(alpha * h1,h1)

        # Logits and tanh output
        logits = tf.layer.dense(h1,out_dim, activation=None)
        out = tf.tanh(logits)

        return out
