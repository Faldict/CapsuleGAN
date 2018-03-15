import numpy as np
import tensorflow as tf
import capsule


def capsule_discriminator(x_image):
    """ capsule network as discriminator
    """
    x = tf.reshape(x_image, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=256,
        kernel_size=[9, 9],
        padding="valid",
        activation=tf.nn.relu,
        name="ReLU_Conv1")
    conv1 = tf.expand_dim(conv1, axis=-2)
    # Convolutional capsules
    primary_caps = capsule.conv2d(conv1, 32, 8, [9, 9], strides=(2, 2), name="PrimaryCaps")
    primary_caps = tf.reshape(primary_caps, [-1, primary_caps.shape[1].value * primary_caps.shape[2].value * 32, 8])
    # Fully Connected capsules with routing by agreement. Binary classifier.
    digit_caps = capsule.dense(primary_caps, 2, 16, iter_routing=3, learn_coupling=False,  mapfn_parallel_iterations=16, name="DigitCaps")
    # The lengths of the capsule activation vectors.
    lengths = tf.sqrt(tf.reduce_sum(tf.square(digit_caps), axis=1), name="Lengths")
    return lengths




