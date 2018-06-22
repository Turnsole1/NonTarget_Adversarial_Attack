"""Implementation of sample attack. Finally, the generated adversarial sample is judged by eight models

        noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
        noise = momentum * grad + noise
        adv = x + alpha * tf.sign(noise)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave

import tensorflow as tf
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2
import time

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_v3', './models/inception_v3.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_adv_inception_v3', './models/adv_inception_v3_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens3_adv_inception_v3', './models/ens3_adv_inception_v3_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens4_adv_inception_v3', './models/ens4_adv_inception_v3_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_v4', './models/inception_v4.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_resnet_v2', './models/inception_resnet_v2_2016_08_30.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens_adv_inception_resnet_v2', './models/ens_adv_inception_resnet_v2_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_resnet', './models/resnet_v2_101.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', './adv_samples', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', './out_images_new_v1', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'num_iter', 10, 'Number of iterations.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 2, 'How many images process at one time.')

tf.flags.DEFINE_float(
    'momentum', 1.0, 'Momentum.')

tf.flags.DEFINE_string(
    'GPU_ID', '2', 'which GPU to use.')

FLAGS = tf.flags.FLAGS

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.GPU_ID

def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.
    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1

        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.
    Args:
      images: array with minibatch of images
      filenames: list of filenames without path
        If number of file names in this list less than number of images in
        the minibatch then only first len(filenames) images will be saved.
      output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')

def main(_):
    start = time.clock()
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    # eps = 2.0 * FLAGS.max_epsilon / 255.0
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    tf.logging.set_verbosity(tf.logging.INFO)

    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum
    num_classes = 1001

    with tf.Session() as sess:
        x = tf.placeholder(dtype=tf.float32, shape=batch_shape)

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_v3, end_points_v3 = inception_v3.inception_v3(
                x, num_classes=num_classes, is_training=False)
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_adv_v3, end_points_adv_v3 = inception_v3.inception_v3(
                x, num_classes=num_classes, is_training=False, scope='AdvInceptionV3')
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
                x, num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3')
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
                x, num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')
        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits_v4, end_points_v4 = inception_v4.inception_v4(
                x, num_classes=num_classes, is_training=False)
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
                x, num_classes=num_classes, is_training=False)
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_resnet_v2.inception_resnet_v2(
                x, num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2')
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits_resnet, end_points_resnet = resnet_v2.resnet_v2_101(
                x, num_classes=num_classes, is_training=False)

        pred = tf.argmax(
            end_points_v3['Predictions'] + end_points_adv_v3['Predictions'] + end_points_ens3_adv_v3['Predictions'] + \
            end_points_ens4_adv_v3['Predictions'] + end_points_v4['Predictions'] + \
            end_points_res_v2['Predictions'] + end_points_ensadv_res_v2['Predictions'] + end_points_resnet['predictions'], 1)


        y = tf.placeholder(tf.int32, shape=[FLAGS.batch_size,])
        one_hot = tf.one_hot(y, num_classes)

        logits_dict = [logits_v3, logits_adv_v3, logits_ens3_adv_v3, logits_ens4_adv_v3, logits_v4, logits_res_v2, logits_ensadv_res_v2, logits_resnet]

        logits_gravity = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, 8])

        logits0 = (logits_dict[0][0] * logits_gravity[0][0] + logits_dict[1][0] * logits_gravity[0][1] + logits_dict[2][0] * logits_gravity[0][2]\
                  + logits_dict[3][0] * logits_gravity[0][3] + logits_dict[4][0] * logits_gravity[0][4] + logits_dict[5][0] * logits_gravity[0][5]\
                  + logits_dict[6][0] * logits_gravity[0][6] + logits_dict[7][0] * logits_gravity[0][7]) / 36

        logits1 = (logits_dict[0][1] * logits_gravity[1][0] + logits_dict[1][1] * logits_gravity[1][1] + logits_dict[2][1] * logits_gravity[1][2]\
                  + logits_dict[3][1] * logits_gravity[1][3] + logits_dict[4][1] * logits_gravity[1][4] + logits_dict[5][1] * logits_gravity[1][5]\
                  + logits_dict[6][1] * logits_gravity[1][6] + logits_dict[7][1] * logits_gravity[1][7]) / 36

        logits0 = tf.reshape(logits0, [1, 1001])
        logits1 = tf.reshape(logits1, [1, 1001])
        logits = tf.concat([logits0, logits1], 0)

        cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                        logits,
                                                        label_smoothing=0.0,
                                                        weights=1.0)

        noise = tf.gradients(cross_entropy, x)[0]
        noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)   # 可以改成 noise = noise / tf.reduce_sum(tf.abs(noise), [1, 2, 3], keep_dims=True)
        grad = tf.placeholder(tf.float32, shape=batch_shape)
        noise = momentum * grad + noise
        adv = x + alpha * tf.sign(noise)
        x_max = tf.placeholder(tf.float32, shape=batch_shape)
        x_min = tf.placeholder(tf.float32, shape=batch_shape)
        adv = tf.clip_by_value(adv, x_min, x_max)






        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        s4 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        s5 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s7 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        s8 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))


        s1.restore(sess, FLAGS.checkpoint_path_inception_v3)
        s2.restore(sess, FLAGS.checkpoint_path_adv_inception_v3)
        s3.restore(sess, FLAGS.checkpoint_path_ens3_adv_inception_v3)
        s4.restore(sess, FLAGS.checkpoint_path_ens4_adv_inception_v3)
        s5.restore(sess, FLAGS.checkpoint_path_inception_v4)
        s6.restore(sess, FLAGS.checkpoint_path_inception_resnet_v2)
        s7.restore(sess, FLAGS.checkpoint_path_ens_adv_inception_resnet_v2)
        s8.restore(sess, FLAGS.checkpoint_path_resnet)

        sum = 0
        failure_num = 0
        l2_distance = 0
        label_distance = 0
        images = []

        for filenames, images in load_images(FLAGS.input_dir, batch_shape):
            images = images.astype(np.float32)               #不知道需不需要!!!!!!!!!!!!!!!!!!!!!!!
            images_flatten_initial = images.reshape((2, 268203))
            sum += len(filenames)
            # 对于每个图片在迭代生成对抗样本的过程中，x_max和x_min 是不变的！！！！！！！
            x_Max = np.clip(images + eps, -1.0, 1.0)
            x_Min = np.clip(images - eps, -1.0, 1.0)

            prediction = []
            Noise = []

            for i in range(FLAGS.num_iter):
                if i == 0:
                    prediction = sess.run(pred, feed_dict={x: images})
                    print('true_label::::::::', prediction)

                # x可能有问题!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                End_points_v3, End_points_adv_v3, End_points_ens3_adv_v3, End_points_ens4_adv_v3, End_points_v4, End_points_res_v2, End_points_ensadv_res_v2, End_points_resnet = \
                sess.run([end_points_v3, end_points_adv_v3, end_points_ens3_adv_v3, end_points_ens4_adv_v3, end_points_v4, end_points_res_v2, end_points_ensadv_res_v2, end_points_resnet], feed_dict={x: images, y: prediction})

                logits_proportion = []
                for j in range(FLAGS.batch_size):
                    end_points_v3_Pred = End_points_v3['Predictions'][j][prediction[j]]
                    end_points_adv_v3_Pred = End_points_adv_v3['Predictions'][j][prediction[j]]
                    end_points_ens3_adv_v3_Pred = End_points_ens3_adv_v3['Predictions'][j][prediction[j]]
                    end_points_ens4_adv_v3_Pred = End_points_ens4_adv_v3['Predictions'][j][prediction[j]]
                    end_points_v4_Pred = End_points_v4['Predictions'][j][prediction[j]]
                    end_points_res_v2_Pred = End_points_res_v2['Predictions'][j][prediction[j]]
                    end_points_ensadv_res_v2_Pred = End_points_ensadv_res_v2['Predictions'][j][prediction[j]]
                    end_points_resnet_Pred = End_points_resnet['predictions'][j][prediction[j]]

                    print('end_points_v3_Pred::::::', end_points_v3_Pred)
                    print('end_points_adv_v3_Pred::::::', end_points_adv_v3_Pred)
                    print('end_points_ens3_adv_v3_Pred::::::', end_points_ens3_adv_v3_Pred)
                    print('end_points_ens4_adv_v3_Pred::::::', end_points_ens4_adv_v3_Pred)
                    print('end_points_v4_Pred::::::', end_points_v4_Pred)
                    print('end_points_res_v2_Pred::::::', end_points_res_v2_Pred)
                    print('end_points_ensadv_res_v2_Pred::::::', end_points_ensadv_res_v2_Pred)
                    print('end_points_resnet_Pred::::::', end_points_resnet_Pred)


                    ens_Pred_Value = np.array([end_points_v3_Pred, end_points_adv_v3_Pred, end_points_ens3_adv_v3_Pred,
                                      end_points_ens4_adv_v3_Pred, end_points_v4_Pred, end_points_res_v2_Pred,
                                      end_points_ensadv_res_v2_Pred, end_points_resnet_Pred])
                    print('ens_Pred_Value:::::', ens_Pred_Value)
                    TopKFitIndx = np.argsort(ens_Pred_Value)

                    a = [0.0] * 8

                    for m in range(8):
                        a[TopKFitIndx[m]] = 8 - m
                        # a[m] = 1
                    logits_proportion.append(a)

                if i == 0:
                    Grad = np.zeros(shape=[FLAGS.batch_size, 299, 299, 3], dtype= np.float32)
                    Noise, images = sess.run([noise, adv], feed_dict={x: images, y: prediction, logits_gravity: logits_proportion, grad: Grad, x_max: x_Max, x_min: x_Min})   # x可能有问题!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                else:
                    Noise, images = sess.run([noise, adv], feed_dict={x: images, y: prediction, logits_gravity: logits_proportion, grad: Noise, x_max: x_Max, x_min: x_Min})   # Noise可能有问题!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


            print('images::::::', images)
            adv_prediction = sess.run(pred, feed_dict={x: images})
            images_flatten_adv = images.reshape((2, 268203))
            save_images(images, filenames, FLAGS.output_dir)

            l2_diatance_list = np.linalg.norm((images_flatten_initial - images_flatten_adv), axis=1, keepdims=True)
            for n in range(len(filenames)):
                l2_distance += l2_diatance_list[n]
            for j in range(len(filenames)):
                label_distance += abs(prediction[j] - adv_prediction[j])
                if int(prediction[j]) == adv_prediction[j]:
                    failure_num += 1
                    print('failure_num:::::', failure_num)
            print('Prediction:::::::', adv_prediction)


        print('sum::::', sum)
        print('failure_num::::', failure_num)
        rate_wrong = failure_num / sum
        print('rate_wrong:::::::', rate_wrong)
        print('l2_distance:::::::', l2_distance)
        print('label_distance::::::', label_distance)
        end = time.clock()
        print('run time::::::::', end - start)


if __name__ == '__main__':
  tf.app.run()