import tensorflow as tf
import numpy as np
import os
import random
from PIL import Image
from PIL import ImageFile
import mynet
# from scipy import misc
from datetime import datetime
import time
import tools
from PerceNet import Vgg16

ImageFile.LOAD_TRUNCATED_IMAGES = True
train_epochs = 100

INPUT_HEIGHT = 224
INPUT_WIDTH = 224

batch_size = 35


checkpoint_path = './log/3.14/'


if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)


def total_variation_loss(inputs):
    """
    A smooth loss in fact. Like the smooth prior in MRF.
    V(y) = || y_{n+1} - y_n ||_2
    :param inputs:
    :return:
    """
    dy = inputs[:, :-1, ...] - inputs[:, 1:, ...]
    dx = inputs[:, :, :-1, ...] - inputs[:, :, 1:, ...]
    size_dy = tf.size(dy, out_type=tf.float32)
    size_dx = tf.size(dx, out_type=tf.float32)
    return tf.nn.l2_loss(dy) / size_dy + tf.nn.l2_loss(dx) / size_dx


def get_train_batch(pointer):
    image_batch = []
    images = val_images[pointer * batch_size:(pointer + 1) * batch_size]

    for img in images:
        arr = Image.open(img)
        arr = arr.crop((100, 100, 324, 324))
        arr = np.array(arr)
        # arr = tf.image.resize_images(arr, (224, 224))
        arr = arr.astype('float32') / 255
        image_batch.append(arr)

    return image_batch


def get_test_batch(pointer):
    image_batch = []
    images = val_hazy[pointer * batch_size:(pointer + 1) * batch_size]
    for img in images:
        arr = Image.open(img)
        # arr = arr.resize((224, 224), resample=Image.BICUBIC)
        arr = arr.crop((100, 100, 324, 324))
        arr = np.array(arr)
        # arr = tf.image.resize_images(arr, (224, 224))
        arr = arr.astype('float32') / 255
        # arr = np.resize(arr, [224, 224])
        image_batch.append(arr)
    return image_batch


def get_batches(pointer, temp):
    clear_batch = []
    hazy_batch = []

    # temp = [train_images, test_images]
    # random.shuffle(temp)
    image_1 = temp[0]
    image_2 = temp[1]
    for img in image_1[pointer * batch_size:(pointer + 1) * batch_size]:
        arr = Image.open(img)
        # arr = arr.resize((224, 224), resample=Image.BICUBIC)
        arr = arr.crop((100, 100, 324, 324))
        arr = np.array(arr)
        # arr = tf.image.resize_images(arr, (224, 224))
        arr = arr.astype('float32') / 255
        # arr = np.resize(arr, [224, 224])
        clear_batch.append(arr)
    for img in image_2[pointer * batch_size:(pointer + 1) * batch_size]:
        arr = Image.open(img)
        # arr = arr.resize((224, 224), resample=Image.BICUBIC)
        arr = arr.crop((100, 100, 324, 324))
        arr = np.array(arr)
        # arr = tf.image.resize_images(arr, (224, 224))
        arr = arr.astype('float32') / 255
        # arr = np.resize(arr, [224, 224])
        hazy_batch.append(arr)

    return clear_batch, hazy_batch

IMAGE_DATE_DIR = './clear1/'
IMAGE_DATE_DIR1 = './OTS/'
Val_Dir = './gt/'
Val_Hazy_Dir = './hazy/'


train_images = []
for image_filename in os.listdir(IMAGE_DATE_DIR):
    if image_filename.endswith('.jpg'):
        for _ in range(35):
            train_images.append(os.path.join(IMAGE_DATE_DIR, image_filename))

# random.shuffle(train_images)

test_images = []
for image_filename in os.listdir(IMAGE_DATE_DIR1):
    if image_filename.endswith('.jpg'):
        test_images.append(os.path.join(IMAGE_DATE_DIR1, image_filename))

# random.shuffle(test_images)

val_images = []
for image_filename in os.listdir(Val_Dir):
    if image_filename.endswith('.png'):
        val_images.append(os.path.join(Val_Dir, image_filename))

val_hazy = []
for image_filename in os.listdir(Val_Hazy_Dir):
    if image_filename.endswith('.jpg'):
        val_hazy.append(os.path.join(Val_Hazy_Dir, image_filename))


input_x = tf.placeholder(tf.float32, shape=[batch_size, INPUT_HEIGHT, INPUT_WIDTH, 3])
input_raw = tf.placeholder(tf.float32, shape=[batch_size, INPUT_HEIGHT, INPUT_WIDTH, 3])

output = mynet.DehazeNet(input_x)
print(np.shape(output))

# vgg_per = Vgg16()
# vgg_per.build(output)
# vgg_tru = Vgg16()
# vgg_tru.build(input_raw)
# output_per = vgg_per.conv3_3
# output_tru = vgg_tru.conv3_3

# loss_tv

loss = tf.reduce_mean(tf.square(tf.subtract(output, input_raw)))

optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

saver = tf.train.Saver()

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

    print('batch size: %d' % batch_size)

    total_batch = int(52500 / batch_size)
    print('total batchs: %d' % total_batch)

    sess.run(init)

    for epoch in range(train_epochs):

        temp = list(zip(train_images, test_images))
        random.shuffle(temp)
        A = [e[0] for e in temp]
        B = [e[1] for e in temp]
        temp_new = [A, B]
        # random.shuffle(train_images)

        # temp_val = list(zip(val_images, val_hazy))
        # random.shuffle(temp_val)
        # C = [e[0] for e in temp_val]
        # D = [e[1] for e in temp_val]
        # temp_new_val = [C, D]

        for batch_index in range(total_batch):
            # start_time = time.time()
            # batch_x = get_train_batch(batch_index)
            # hazy_x = get_test_batch(batch_index)

            batch_x, hazy_x = get_batches(batch_index, temp_new)

            _, train_loss, pred_result = sess.run([optimizer, loss, output], feed_dict={input_x: hazy_x,
                                                                                        input_raw: batch_x})
            print('epoch: %04d\tbatch: %04d\ttrain loss: %.9f' % (epoch + 1, batch_index + 1, train_loss,))

            psnr_train = 0
            for index in range(batch_size):
                array = np.reshape(pred_result[index], newshape=[INPUT_HEIGHT, INPUT_WIDTH, 3])
                array = array * 255
                arr1 = np.uint8(array)
                image = Image.fromarray(arr1, 'RGB')
                # if image.mode != 'L':
                #     image = image.convert('L')
                # image.save('./pred3_10//' + str(epoch) + '_' + str(batch_index) + '.jpg', 'jpeg')

                # misc.imsave('./pred1_26/' + str(epoch * batch_size + index) + '.jpg', arr1)

                array_raw = np.reshape(hazy_x[index], newshape=[INPUT_HEIGHT, INPUT_WIDTH, 3])
                array_raw = array_raw * 255
                arr2 = np.uint8(array_raw)
                image_raw = Image.fromarray(arr2, 'RGB')
                # if image_raw.mode != 'L':
                #     image_raw = image_raw.convert('L')
                # image_raw.save('./pred3_10//' + str(epoch) + '_' + str(batch_index) + '_hazy.jpg', 'jpeg')

                array_truth = np.reshape(batch_x[index], newshape=[INPUT_HEIGHT, INPUT_WIDTH, 3])
                array_truth = array_truth * 255
                arr3 = np.uint8(array_truth)
                image_truth = Image.fromarray(arr3, 'RGB')
                # if image.mode != 'L':
                #     image = image.convert('L')
                # image_truth.save('./pred3_10//' + str(epoch) + '_' + str(batch_index) + '_truth.jpg', 'jpeg')

                psnr_train_per = tools.cal_psnr(arr1, arr3)
                psnr_train += psnr_train_per
            psnr_train_avg = psnr_train / 35
            print('psnr train: %.6f' % psnr_train_avg)


                # duration = time.time() - start_time
                # print(duration)
        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
        saver.save(sess, checkpoint_name)

        total_batch_val = int(490 / 35)
        psnr_all = 0
        for i in range(total_batch_val):

            val_y = get_train_batch(i)
            hazy_y = get_test_batch(i)

            for j in range(35):

                full_path = tf.train.latest_checkpoint(checkpoint_path)
                saver.restore(sess, full_path)

                val_loss, val_result = sess.run([loss, output], feed_dict={input_x: hazy_y,
                                                                           input_raw: val_y})

                array = np.reshape(val_result[j], newshape=[INPUT_HEIGHT, INPUT_WIDTH, 3])
                array = array * 255
                arr1 = np.uint8(array)
                # image = Image.fromarray(arr1, 'RGB')
                # if image.mode != 'L':
                #     image = image.convert('L')
                # image.save('./pred3_12_val//' + str(epoch) + '_' + str(i) + '_' + str(j) + '.jpg', 'jpeg')

                array_truth = np.reshape(val_y[j], newshape=[INPUT_HEIGHT, INPUT_WIDTH, 3])
                array_truth = array_truth * 255
                arr3 = np.uint8(array_truth)
                # image_truth = Image.fromarray(arr3, 'RGB')
                # if image.mode != 'L':
                #     image = image.convert('L')
                # image_truth.save('./pred3_12_val//' + str(epoch) + '_' + str(i) + '_' + str(j) + '_truth.jpg', 'jpeg')

                psnr_val = tools.cal_psnr(arr1, arr3)
                print('epoch: %04d\tper psnr: %.9f' % (epoch + 1, psnr_val))
                psnr_all += psnr_val
        psnr_avg = psnr_all / 490
        print('psnr val: %.6f' % psnr_avg)
        print('End val')
