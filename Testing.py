import tensorflow as tf
import numpy as np
import os
from PIL import Image
import mynet
import tools
from skimage import measure
from scipy import misc
import cv2
import time


test_dir = './out_test_hazy/'
test_tru_dir = './out_test_gt/'


def DehazeNet(x, h, w):
    with tf.variable_scope('DehazeNet'):
        x_s = x
        # x = tools.conv('DN_conv1_1', x_s, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        # x = tools.conv('DN_conv1_2', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

        # with tf.name_scope('pool1'):
        #     x = tools.pool('pool1', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        #
        # with tf.name_scope('pool2'):
        #     x = tools.pool('pool2', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = tools.conv('upsampling_1', x_s, 128, kernel_size=[3, 3], stride=[1, 2, 2, 1])
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

        x = tools.deconv('DN_deconv1', x, 64, output_shape=[1, int((h + 1)/2), int((w + 1)/2), 64], kernel_size=[3, 3],
                         stride=[1, 2, 2, 1])
        x = tools.deconv('DN_deconv2', x, 64, output_shape=[1, h, w, 64], kernel_size=[3, 3], stride=[1, 2, 2, 1])

        x_r = tools.conv_nonacti('DN_conv7_1', x, 3, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x_r = tf.add(x_r, x_s)
        x_r = tools.acti_layer(x_r)
        return x_r


def get_train_batch(pointer):
    image_batch = []
    shape_truth = []
    images = test_images[pointer * 1:(pointer + 1) * 1]

    for img in images:
        arr = Image.open(img)
        # arr = arr.crop((100, 100, 324, 324))
        shape_truth.append(np.size(arr))
        # arr = arr.resize((224, 224))
        arr = np.array(arr)
        # arr = tf.image.resize_images(arr, (224, 224))
        arr = arr.astype('float32') / 255
        image_batch.append(arr)

    return image_batch, shape_truth


def get_test_batch(pointer):
    image_batch = []
    shape_hazy = []
    images = test_hazy[pointer * 1:(pointer + 1) * 1]
    for img in images:
        arr = Image.open(img)
        # arr = arr.crop((100, 100, 324, 324))
        shape_hazy.append(np.size(arr))
        # arr = arr.resize((224, 224))
        arr = np.array(arr)
        # arr = tf.image.resize_images(arr, (224, 224))
        arr = arr.astype('float32') / 255
        # arr = np.resize(arr, [224, 224])
        image_batch.append(arr)
    image_batch = np.expand_dims(image_batch, axis=0)
    return image_batch, shape_hazy

test_images = []
for image_filename in os.listdir(test_dir):
    name_spt = image_filename[0:(len(image_filename) - 4)].split('_')
    for name in os.listdir(test_tru_dir):
        if name.endswith('.png') and name[0:(len(name) - 4)] == name_spt[0]:
            test_images.append(os.path.join(test_tru_dir, name))

print(len(test_images))

test_hazy = []
for image_filename in os.listdir(test_dir):
    if image_filename.endswith('.png'):
        test_hazy.append(os.path.join(test_dir, image_filename))

print(len(test_hazy))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

init = tf.global_variables_initializer()

sess.run(init)

psnr_total = 0
ssim_total = 0

for i in range(500):

    test_y, shape_t = get_train_batch(i)
    hazy_y, shape_h = get_test_batch(i)
    for j in range(1):

        input_x = tf.placeholder(tf.float32, shape=[1, shape_h[j][1], shape_h[j][0], 3])
        input_raw = tf.placeholder(tf.float32, shape=[1, shape_t[j][1], shape_t[j][0], 3])

        output = DehazeNet(input_x, shape_h[j][1], shape_h[j][0])
        loss = tf.reduce_mean(tf.square(tf.subtract(output, input_raw)))
        saver = tf.train.Saver()

        start = time.time()

        ckpt = tf.train.get_checkpoint_state("./log/11_28_outdoor/")
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('====================================')
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            # saver_new.restore(sess, './log/3.15/model_epoch1.ckpt')

        duration = time.time()-start
        print(duration)

        val_result = sess.run(output, feed_dict={input_x: hazy_y[j]})
        # if i%10 == 0:
        array = np.reshape(val_result[j], newshape=[shape_h[j][1], shape_h[j][0], 3])
        array = array * 255
        array = tf.saturate_cast(array, dtype=tf.uint8)
        arr1 = sess.run(array)
        image = Image.fromarray(arr1, 'RGB')
        # image = image.resize(shape_1[j])
        # image.save('./out_result//' + str(i) + '_' + str(j) + '.png')

        array_hazy = np.reshape(hazy_y[j], newshape=[shape_h[j][1], shape_h[j][0], 3])
        array_hazy = array_hazy * 255
        array_hazy = tf.saturate_cast(array_hazy, dtype=tf.uint8)
        arr2 = sess.run(array_hazy)
        image_hazy = Image.fromarray(arr2, 'RGB')
        # image_hazy = image_hazy.resize(shape_2[j])
        # image_hazy.save('./out_result//' + str(i) + '_' + str(j) + '_hazy.png')

        truth = np.reshape(test_y[j], newshape=[shape_h[j][1], shape_h[j][0], 3])
        truth = truth * 255
        arr3 = np.uint8(truth)
        image_truth = Image.fromarray(arr3, 'RGB')

        psnr_test = tools.cal_psnr(arr1, arr3)
        ssim_test = measure.compare_ssim(arr1, arr3, multichannel=True)
        print('No: %04d\tsingle psnr: %.6f\tsingle ssim: %.6f' % (i + 1, psnr_test, ssim_test))
        psnr_total += psnr_test
        ssim_total += ssim_test

psnr_average = psnr_total / 500
ssim_average = ssim_total / 500
print('psnr val: %.6f\tssim val: %.6f' % (psnr_average, ssim_average))
print('End val')
