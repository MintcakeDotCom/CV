import tensorflow as tf
import numpy as np
import os
from PIL import Image
import mynet
import tools


test_dir = './hazy/'
test_tru_dir = './gt/'


def get_train_batch(pointer):
    image_batch = []
    images = test_images[pointer * 35:(pointer + 1) * 35]

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
    images = test_hazy[pointer * 35:(pointer + 1) * 35]
    for img in images:
        arr = Image.open(img)
        arr = arr.crop((100, 100, 324, 324))
        arr = np.array(arr)
        # arr = tf.image.resize_images(arr, (224, 224))
        arr = arr.astype('float32') / 255
        # arr = np.resize(arr, [224, 224])
        image_batch.append(arr)
    return image_batch

test_images = []
for image_filename in os.listdir(test_tru_dir):
    if image_filename.endswith('.png'):
        test_images.append(os.path.join(test_tru_dir, image_filename))

test_hazy = []
for image_filename in os.listdir(test_dir):
    if image_filename.endswith('.jpg'):
        test_hazy.append(os.path.join(test_dir, image_filename))


input_x = tf.placeholder(tf.float32, shape=[35, 224, 224, 3])
input_raw = tf.placeholder(tf.float32, shape=[35, 224, 224, 3])

output = mynet.DehazeNet(input_x)
loss = tf.reduce_mean(tf.square(tf.subtract(output, input_raw)))

# init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


with tf.Session(config=config) as sess:

    # sess.run(init)
    saver = tf.train.import_meta_graph('./log/3.11/model_epoch17.ckpt.meta')
    saver.restore(sess, './log/3.11/model_epoch17.ckpt')

    total_batch = int(490 / 35)
    psnr_total = 0
    for i in range(total_batch):

        test_y = get_train_batch(i)
        hazy_y = get_test_batch(i)

        val_loss, val_result = sess.run([loss, output], feed_dict={input_x: hazy_y,
                                                                   input_raw: test_y})

        for j in range(35):

            array = np.reshape(val_result[j], newshape=[224, 224, 3])
            array = array * 255
            arr1 = np.uint8(array)
            image = Image.fromarray(array, 'RGB')
            image.save('./testresults//' + str(i) + '_' + str(j) + '.jpg', 'jpeg')

            array_hazy = np.reshape(hazy_y[j], newshape=[224, 224, 3])
            array_hazy = array_hazy * 255
            arr2 = np.uint8(array_hazy)
            image_hazy = Image.fromarray(arr2, 'RGB')
            image_hazy.save('./testresults//' + str(i) + '_' + str(j) + '_hazy.jpg', 'jpeg')

            truth = np.reshape(test_y[j], newshape=[224, 224, 3])
            truth = truth * 255
            arr3 = np.uint8(truth)

            psnr_test = tools.cal_psnr(arr1, arr3)
            print('batch: %04d\tNo: %04d\tsingle psnr: %.9f' % (i + 1, j + 1, psnr_test))
            psnr_total += psnr_test

    psnr_average = psnr_total / 490
    print('psnr val: %.6f' % psnr_average)
    print('End val')
