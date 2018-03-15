import tensorflow as tf
import numpy as np
import tools


def DehazeNet(x):

    with tf.name_scope('DehazeNet'):

        x_s = tools.conv('DN_conv1_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv('DN_conv1_2', x_s, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

        # with tf.name_scope('pool1'):
        #     x = tools.pool('pool1', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        #
        # with tf.name_scope('pool2'):
        #     x = tools.pool('pool2', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = tools.conv('upsampling_1', x, 128, kernel_size=[3, 3], stride=[1, 2, 2, 1])
        x = tools.conv('upsampling_2', x, 128, kernel_size=[3, 3], stride=[1, 2, 2, 1])

        x1 = tools.conv('DN_conv1_3', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv('DN_conv2_1', x1, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv_nonacti('DN_conv2_2', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tf.add(x, x1)
        # x = tools.batch_norm(x)
        x = tools.acti_layer(x)

        # x = tools.conv('DN_conv2_3', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv('DN_conv2_4', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

        x2 = tools.conv('DN_conv3_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv('DN_conv3_2', x2, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv_nonacti('DN_conv3_3', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tf.add(x, x2)
        # x = tools.batch_norm(x)
        x = tools.acti_layer(x)

        # x = tools.conv('DN_conv3_4', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv('DN_conv3_5', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

        x3 = tools.conv('DN_conv4_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv('DN_conv4_2', x3, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv('DN_conv4_3', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv_nonacti('DN_conv4_4', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tf.add(x, x3)
        # x = tools.batch_norm(x)
        x = tools.acti_layer(x)

        x = tools.conv('DN_conv4_5', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

        x4 = tools.conv('DN_conv5_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv('DN_conv5_2', x4, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv('DN_conv5_3', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv('DN_conv5_4', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv_nonacti('DN_conv5_5', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tf.add(x, x4)
        # x = tools.batch_norm(x)
        x = tools.acti_layer(x)

        x = tools.deconv('DN_deconv1', x, 64, output_shape=[35, 112, 112, 64], kernel_size=[3, 3], stride=[1, 2, 2, 1])

        # x5 = tools.conv('DN_conv6_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        # x = tools.conv('DN_conv6_2', x5, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        # x = tools.conv_nonacti('DN_conv6_3', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        # x = tf.add(x, x5)
        # # x = tools.batch_norm(x)
        # x = tools.acti_layer(x)

        x = tools.deconv('DN_deconv2', x, 64, output_shape=[35, 224, 224, 64], kernel_size=[3, 3], stride=[1, 2, 2, 1])
        x = tools.conv('DN_conv6_6', x, 3, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv_nonacti('DN_conv6_7', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        # x6 = tools.conv('DN_conv6_4', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        # x = tools.conv('DN_conv6_5', x6, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        # x = tools.conv_nonacti('DN_conv6_6', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tf.add(x, x_s)
        # # x = tools.batch_norm(x)
        x = tools.acti_layer(x)

        x = tools.conv('DN_conv6_8', x, 3, kernel_size=[3, 3], stride=[1, 1, 1, 1])

        # x = tools.conv('conv6_4', x, 3, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        # x = tools.FC_layer('fc6', x, out_nodes=4096)
        # with tf.name_scope('batch_norm1'):

        #     x = tools.batch_norm(x)
        # x = tools.FC_layer('fc7', x, out_nod
        #
        # es=4096)
        # with tf.name_scope('batch_norm2'):
        #     x = tools.batch_norm(x)
        # x = tools.FC_layer('fc8', x, out_nodes=n_classes)

        return x

        # with tf.name_scope('pool3'):
        #     x = tools.pool('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        # with tf.name_scope('pool4'):
        #     x = tools.pool('pool4', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        # with tf.name_scope('pool5'):
        #     x = tools.pool('pool5', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
