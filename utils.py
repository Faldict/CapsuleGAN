import numpy as np
import tensorflow as tf

def margin_loss(onehot_labels, lengths, m_plus=0.9, m_minus=0.1, l=0.5):
    T = tf.to_float(onehot_labels)
    lengths = tf.to_float(lengths)
    L_present = T * tf.square(tf.nn.relu(m_plus - lengths))
    L_absent = (1 - T) * tf.square(tf.nn.relu(lengths - m_minus))
    L = L_present + l * L_absent
    return tf.reduce_mean(L)