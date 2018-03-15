import numpy as np 
import tensorflow as tf
from discriminator import *
from generator import generator
from utils import margin_loss


tf.logging.set_verbosity(tf.logging.INFO)
mnist = tf.contrib.learn.datasets.load_dataset("mnist")

lr = 1e-3
batch_size = 32
z_dim = 100
max_epochs = 10000
d_step = 50
g_step = 50

# mnist image data
x_placeholder = tf.placeholder("float", shape=[batch_size, 28, 28, 1], name="x_placeholder")

Gz = generator(batch_size, z_dim)
Dx = capsule_discriminator(x_placeholder)
Dg = capsule_discriminator(Gz)

# loss function
g_loss = margin_loss(1, Dg)
d_loss_real = margin_loss(1, Dx)
d_loss_fake = margin_loss(0, Dg)
d_loss = d_loss_real + d_loss_fake

thetas = tf.trainable_variables()
theta_d = [var for var in thetes if 'd_' in var.name]
theta_g = [var for var in thetas if 'g_' in var.name]

d_solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(d_loss, var_list=theta_d)
g_solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(g_loss, var_list=theta_g)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(max_epochs):
        x_image = mnist.train.next_batch(batch_size)

        for step in range(d_step):
            d_loss_cur, _ = sess.run([d_loss, d_solver], feed_dict={x_placeholder: x_image})

        for step in range(g_step):
            g_loss_cur, _ = sess.run([g_loss, g_solver], feed_dict={x_placeholder: x_image})
