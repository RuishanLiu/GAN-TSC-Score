"""
Code modified from: https://github.com/chengshengchan/model_compression
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, time

def get_compression_score(data, path_teacher='compression_teacher.npy', n_iter=3,  retrain_real=False, gpu=0):
    '''
    Calucate Compression Score for CIFAR-10
    - Input
        data: images we want to examine
        path_teacher: path to teacher weights .npy
        n_iter: number of independent runs
        retrain_real: whether to retrain on the real training data as baseline
        gpu: which gpu to use
    - Output
        compression score: (mean, std)
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not os.path.isfile(path_teacher):
        sys.exit("ERROR: Path to compression_teacher.npy file is not valid. Make sure the .npy file is included. \nInclude .npy file or change path_teacher. Default - compression_score(data, path_teacher='compression_teacher.npy')")
    acc, acc_var = get_compression_acc(data, path_teacher=path_teacher, n_iter=n_iter)
    if retrain_real:
        (data_real, _), (_, _) = tf.keras.datasets.cifar10.load_data()
        acc_real, _ = get_compression_acc(data_real, path_teacher=path_teacher, n_iter=n_iter)
    else:
        acc_real = 0.532
    return acc/acc_real, acc_var/acc_real
        
def get_compression_acc(data, path_teacher='compression_teacher.npy', n_iter=3):
    # Parameters
    dim = 32
    n_classes = 10
    dropout_rate = 0.5
    batch_size = 256
    lr = 1e-4
    
    # Placeholders
    x = tf.placeholder(tf.float32, [batch_size, dim, dim, 3])
    keep_prob = tf.placeholder(tf.float32) 
    
    # Load Data
    (data_train, label_train), (data_test, label_test) = tf.keras.datasets.cifar10.load_data()
    mean = np.mean(data_train, axis=0)
       
    # Load Model and Basic Settings
    teacher=nin(x, keep_prob, path_teacher)
    student=lenet(x, keep_prob)
    pred = tf.nn.softmax(student)
    tf_loss = tf.nn.l2_loss(teacher - student)/batch_size
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(tf_loss)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
     
    acc = []
    index = np.array(range(len(data))) 
    iterations = int(len(data)/batch_size)
    for i_iter in range(n_iter):
        # Random Seed for Reproduction
        tf.set_random_seed(100+i_iter)
        tf.global_variables_initializer().run()
        with tf.device('/cpu:0'):
            saver = tf.train.Saver(max_to_keep=100)
        # Train    
        for j in xrange(iterations):
            batch_x = data[index[j*batch_size:(j+1)*batch_size]]       
            batch_x = np.float32(batch_x) - mean
            _, cost = sess.run([optimizer, tf_loss],
                               feed_dict={x : batch_x, keep_prob : 1-dropout_rate})      
        # Test
        total = 0
        correct = 0
        iterations_test = int(len(data_test)/batch_size)
        for j in xrange(iterations_test):
            batch_x = data_test[j*batch_size:(j+1)*batch_size] - mean
            prob, cost = sess.run([pred, tf_loss],
                    feed_dict={x : batch_x, keep_prob : 1.0})
            label_batch = label_test[j*batch_size:(j+1)*batch_size].reshape(-1)
            pred_batch = np.array( [np.argmax(prob[i]) for i in range(prob.shape[0])])
            correct += sum(label_batch == pred_batch)
            total += batch_size
        print('%d - Compression accuracy: %.4f' % (i_iter, float(correct)/total))
        acc.append(float(correct)/total)
    return np.mean(acc), np.sqrt(np.var(acc, axis=0)/n_iter)

# function of CNN model reference: https://github.com/aymericdamien/TensorFlow-Examples/
# Create some wrappers for simplicity
def conv(x, W, b, strides=1, padding='SAME'):
    # Conv2D wrapper, with bias and relu activation
    #x = tf.pad(x, paddings, "CONSTANT")
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return x

def maxpool2d(x, k, s, padding='SAME'):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],
                          padding=padding)

def avgpool2d(x, k, s, padding='SAME'):
    # AvgPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],
                          padding=padding)


def nin(x, keep_prob, path_teacher): # modify from model - network in network
    # pre-trained weight
    npyfile = np.load(path_teacher)
    npyfile = npyfile.item()

    weights = {
        'conv1': tf.Variable(npyfile['conv1']['weights'], trainable=False, name = 'conv1_w'),
        'cccp1': tf.Variable(npyfile['cccp1']['weights'], trainable=False, name = 'cccp1_w'),
        'cccp2': tf.Variable(npyfile['cccp2']['weights'], trainable=False, name = 'cccp2_w'),
        'conv2': tf.Variable(npyfile['conv2']['weights'], trainable=False, name = 'conv2_w'),
        'cccp3': tf.Variable(npyfile['cccp3']['weights'], trainable=False, name = 'cccp3_w'),
        'cccp4': tf.Variable(npyfile['cccp4']['weights'], trainable=False, name = 'cccp4_w'),
        'conv3': tf.Variable(npyfile['conv3']['weights'], trainable=False, name = 'conv3_w'),
        'cccp5': tf.Variable(npyfile['cccp5']['weights'], trainable=False, name = 'cccp5_w'),
        'ip1': tf.Variable(npyfile['ip1']['weights'], trainable=False, name = 'ip1_w'),
        'ip2': tf.Variable(npyfile['ip2']['weights'], trainable=False, name = 'ip2_w')
    }

    biases = {
        'conv1': tf.Variable(npyfile['conv1']['biases'], trainable=False, name = 'conv1_b'),
        'cccp1': tf.Variable(npyfile['cccp1']['biases'], trainable=False, name = 'cccp1_b'),
        'cccp2': tf.Variable(npyfile['cccp2']['biases'], trainable=False, name = 'cccp2_b'),
        'conv2': tf.Variable(npyfile['conv2']['biases'], trainable=False, name = 'conv2_b'),
        'cccp3': tf.Variable(npyfile['cccp3']['biases'], trainable=False, name = 'cccp3_b'),
        'cccp4': tf.Variable(npyfile['cccp4']['biases'], trainable=False, name = 'cccp4_b'),
        'conv3': tf.Variable(npyfile['conv3']['biases'], trainable=False, name = 'conv3_b'),
        'cccp5': tf.Variable(npyfile['cccp5']['biases'], trainable=False, name = 'cccp5_b'),
        'ip1': tf.Variable(npyfile['ip1']['biases'], trainable=False, name = 'ip1_b'),
        'ip2': tf.Variable(npyfile['ip2']['biases'], trainable=False, name = 'ip2_b')
    }

    conv1 = conv(x, weights['conv1'], biases['conv1'])
    conv1_relu = tf.nn.relu(conv1)
    cccp1 = conv(conv1_relu, weights['cccp1'], biases['cccp1'])
    cccp1_relu = tf.nn.relu(cccp1)
    cccp2 = conv(cccp1_relu, weights['cccp2'], biases['cccp2'])
    cccp2_relu = tf.nn.relu(cccp2)
    pool1 = maxpool2d(cccp2_relu, k=3, s=2)
    drop3 = tf.nn.dropout(pool1, keep_prob)

    conv2 = conv(drop3, weights['conv2'], biases['conv2'])
    conv2_relu = tf.nn.relu(conv2)
    cccp3 = conv(conv2_relu, weights['cccp3'], biases['cccp3'])
    cccp3_relu = tf.nn.relu(cccp3)
    cccp4 = conv(cccp3_relu, weights['cccp4'], biases['cccp4'])
    cccp4_relu = tf.nn.relu(cccp4)

    pool2 = avgpool2d(cccp4_relu, k=3, s=2)
    drop6 = tf.nn.dropout(pool2, keep_prob)

    conv3 = conv(drop6, weights['conv3'], biases['conv3'])
    conv3_relu = tf.nn.relu(conv3)
    cccp5 = conv(conv3_relu, weights['cccp5'], biases['cccp5'])
    cccp5_relu = tf.nn.relu(cccp5)

    # inner product
    ip1 = tf.reshape(cccp5_relu, [-1, weights['ip1'].get_shape().as_list()[0]])
    ip1 = tf.add(tf.matmul(ip1, weights['ip1']), biases['ip1'])
    ip1_relu = tf.nn.relu(ip1)
    ip2 = tf.add(tf.matmul(ip1_relu, weights['ip2']), biases['ip2'])

    return ip2

def lenet(x, keep_prob): # modify from lenet model
    # Random initialize
    weights = {
        'conv1': tf.get_variable('LN_conv1_w', [5,5,3,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'conv2': tf.get_variable('LN_conv2_w', [5,5,64,128],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'ip1': tf.get_variable('LN_ip1_w', [5*5*128, 1024] , initializer=tf.contrib.layers.xavier_initializer()),
        'ip2': tf.get_variable('LN_ip2_w', [1024,10], initializer=tf.contrib.layers.xavier_initializer())
    }

    biases = {
        'conv1': tf.Variable(tf.random_normal(shape=[64],stddev=0.5), name = 'LN_conv1_b'),
        'conv2': tf.Variable(tf.random_normal(shape=[128],stddev=0.5), name = 'LN_conv2_b'),
        'ip1': tf.Variable(tf.random_normal(shape=[1024],stddev=0.5), name = 'LN_ip1_b'),
        'ip2': tf.Variable(tf.random_normal(shape=[10],stddev=0.5), name = 'LN_ip2_b')
    }

    conv1 = conv(x, weights['conv1'], biases['conv1'],padding='VALID')
    pool1 = maxpool2d(conv1, k=2, s=2)
    conv2 = conv(pool1, weights['conv2'], biases['conv2'], padding='VALID')
    pool2 = maxpool2d(conv2, k=2, s=2,padding='VALID')

    ip1 = tf.reshape(pool2, [-1, weights['ip1'].get_shape().as_list()[0]])
    ip1 = tf.add(tf.matmul(ip1, weights['ip1']), biases['ip1'])
    ip1_relu = tf.nn.relu(ip1)
    ip2 = tf.add(tf.matmul(ip1_relu, weights['ip2']), biases['ip2'])
    return ip2

# if False:
#     (data_train, label_train), (data_test, label_test) = tf.keras.datasets.cifar10.load_data()
#     print(get_compression_score(data_train, gpu=1))
#     print(get_compression_score(data_train, gpu=1))
# else:
#     # data_acgan_all = np.load('/datadrive/model_compression/results/Cifar10/ACGAN_pytorch_no_discriminator/fake_data/10/0.npz')
#     # data_acgan = data_acgan_all['data']
#     # print(get_compression_score(data_acgan))

#     data_acgan_all = np.load('/datadrive/model_compression/results/Cifar10/ACGAN_pytorch_no_discriminator/fake_data/499/0.npz')
#     data_acgan = data_acgan_all['data']
#     start = time.time()
#     print(get_compression_score(data_acgan))
#     end = time.time()
#     print(end - start)