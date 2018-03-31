import tensorflow as tf
import numpy as np
import tools


def DehazeNet(x):

    with tf.name_scope('DehazeNet'):
        x_s = x
        x = tools.conv('DN_conv1_1', x_s, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv('DN_conv1_2', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

        # with tf.name_scope('pool1'):
        #     x = tools.pool('pool1', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        #
        # with tf.name_scope('pool2'):
        #     x = tools.pool('pool2', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = tools.conv('upsampling_1', x, 128, kernel_size=[3, 3], stride=[1, 2, 2, 1])
        x = tools.conv('upsampling_2', x, 128, kernel_size=[3, 3], stride=[1, 2, 2, 1])

        x1 = tools.conv('DN_conv2_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv('DN_conv2_2', x1, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv_nonacti('DN_conv2_3', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tf.add(x, x1)
        # x = tools.batch_norm(x)
        x = tools.acti_layer(x)

        # x = tools.conv('DN_conv2_4', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

        x2 = tools.conv('DN_conv3_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv('DN_conv3_2', x2, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv_nonacti('DN_conv3_3', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tf.add(x, x2)
        # x = tools.batch_norm(x)
        x = tools.acti_layer(x)

        # x = tools.conv('DN_conv3_4', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

        x3 = tools.conv('DN_conv4_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv('DN_conv4_2', x3, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv('DN_conv4_3', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv('DN_conv4_4', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv_nonacti('DN_conv4_5', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tf.add(x, x3)
        # x = tools.batch_norm(x)
        x = tools.acti_layer(x)

        # x = tools.conv('DN_conv4_5', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

        x4 = tools.conv('DN_conv5_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv('DN_conv5_2', x4, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv('DN_conv5_3', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv('DN_conv5_4', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv_nonacti('DN_conv5_5', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tf.add(x, x4)
        # x = tools.batch_norm(x)
        x = tools.acti_layer(x)

        x5 = tools.conv('DN_conv5_6', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv('DN_conv5_7', x5, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv_nonacti('DN_conv5_8', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tf.add(x, x5)
        x = tools.acti_layer(x)

        x = tools.deconv('DN_deconv1', x, 64, output_shape=[35, 112, 112, 64], kernel_size=[3, 3], stride=[1, 2, 2, 1])
        x = tools.deconv('DN_deconv2', x, 64, output_shape=[35, 224, 224, 64], kernel_size=[3, 3], stride=[1, 2, 2, 1])

        # x = tools.conv('DN_conv6_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tools.conv('DN_conv6_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        # x = tf.add(x, x_s)
        # # # x = tools.batch_norm(x)
        # x = tools.acti_layer(x)

        x = tools.conv_nonacti('DN_conv7_1', x, 3, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = tf.add(x, x_s)
        x = tools.acti_layer(x)
        return x


# def DehazeNet_evaluate(x):
#     with tf.name_scope('DehazeNet'):
#         x_s = x
#         x = tools.conv('DN_conv1_1', x_s, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#         x = tools.conv('DN_conv1_2', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#
#         # with tf.name_scope('pool1'):
#         #     x = tools.pool('pool1', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
#         #
#         # with tf.name_scope('pool2'):
#         #     x = tools.pool('pool2', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
#
#         x = tools.conv('upsampling_1', x, 128, kernel_size=[3, 3], stride=[1, 2, 2, 1])
#         x = tools.conv('upsampling_2', x, 128, kernel_size=[3, 3], stride=[1, 2, 2, 1])
#
#         x1 = tools.conv('DN_conv2_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#         x = tools.conv('DN_conv2_2', x1, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#         x = tools.conv_nonacti('DN_conv2_3', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#         x = tf.add(x, x1)
#         # x = tools.batch_norm(x)
#         x = tools.acti_layer(x)
#
#         # x = tools.conv('DN_conv2_4', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#
#         x2 = tools.conv('DN_conv3_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#         x = tools.conv('DN_conv3_2', x2, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#         x = tools.conv_nonacti('DN_conv3_3', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#         x = tf.add(x, x2)
#         # x = tools.batch_norm(x)
#         x = tools.acti_layer(x)
#
#         # x = tools.conv('DN_conv3_4', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#
#         x3 = tools.conv('DN_conv4_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#         x = tools.conv('DN_conv4_2', x3, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#         x = tools.conv('DN_conv4_3', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#         x = tools.conv('DN_conv4_4', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#         x = tools.conv_nonacti('DN_conv4_5', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#         x = tf.add(x, x3)
#         # x = tools.batch_norm(x)
#         x = tools.acti_layer(x)
#
#         # x = tools.conv('DN_conv4_5', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#
#         x4 = tools.conv('DN_conv5_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#         x = tools.conv('DN_conv5_2', x4, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#         x = tools.conv('DN_conv5_3', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#         x = tools.conv('DN_conv5_4', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#         x = tools.conv_nonacti('DN_conv5_5', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#         x = tf.add(x, x4)
#         # x = tools.batch_norm(x)
#         x = tools.acti_layer(x)
#
#         x5 = tools.conv('DN_conv5_6', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#         x = tools.conv('DN_conv5_7', x5, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#         x = tools.conv_nonacti('DN_conv5_8', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#         x = tf.add(x, x5)
#         x = tools.acti_layer(x)
#
#         x = tools.deconv('DN_deconv1', x, 64, output_shape=[35, 112, 112, 64], kernel_size=[3, 3], stride=[1, 2, 2, 1])
#         x = tools.deconv('DN_deconv2', x, 64, output_shape=[35, 224, 224, 64], kernel_size=[3, 3], stride=[1, 2, 2, 1])
#
#         # x = tools.conv('DN_conv6_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#         x = tools.conv('DN_conv6_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#         # x = tf.add(x, x_s)
#         # # # x = tools.batch_norm(x)
#         # x = tools.acti_layer(x)
#
#         x = tools.conv_nonacti('DN_conv7_1', x, 3, kernel_size=[3, 3], stride=[1, 1, 1, 1])
#         x = tf.add(x, x_s)
#         x = tools.acti_layer(x)
#         return x
