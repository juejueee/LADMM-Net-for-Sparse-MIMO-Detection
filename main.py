# The simulation environment is
'''
python 3.7
tensorflow-gpu 1.14.0
numpy 1.16.4
'''

import LADMM_Net as ADMM

import Data_generation as data

from Parameters import *

import os

import datetime
import time as tm

# Choose the GPU device
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICE"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tf.set_random_seed(10)
sess = tf.compat.v1.InteractiveSession()

# Tensorflow placeholders
if mod == 0:
    K = K_orig
    N = N_orig
else:
    K = 2 * K_orig
    N = 2 * N_orig

X = tf.compat.v1.placeholder(tf.float32, shape=[None, K, 1])
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, N, 1])
H = tf.compat.v1.placeholder(tf.float32, shape=[None, N, K])
H_T = tf.compat.v1.placeholder(tf.float32,shape=[None,K, N])

batch_size = tf.shape(X)[0]

# Types of MIMO detector
if detector_name =="ZF":
    pinv_H = tf.matmul(tf.linalg.inv(tf.matmul(H_T, H)), H_T)
    x_hat = tf.matmul(pinv_H, Y)
    x_result = func.demod_projection(x_hat, mod)

if detector_name =="LADMM":
    x_hat, x_result, layer_loss, layer_ser, delta, theta, alpha, beta, gamma = ADMM.ADMM_Net(batch_size, K, N, X, Y, H, H_T)


LOSS, SER = func.loss(X, x_hat, x_result, K, mod)


print('Tx:',K_orig,'\t Rx:',N_orig,'\nModulation',mod,'QAM (BPSK = 0 QAM)\nMIMO detector:',detector_name,'\n\nTrain iteration:',train_iter,'\nTrain batch size:',train_batch_size,'\n')
print('Test iteration:',test_iter,'\nTest batch size:',test_batch_size,'\n')

# Training the Neural Networks
if train_iter > 0 and detector_name !="ZF":
    TOTAL_LOSS = tf.add_n(layer_loss)
    saver = tf.compat.v1.train.Saver()

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.compat.v1.train.exponential_decay(startingLearningRate, global_step, decay_step_size, decay_factor, staircase=True)
    train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(TOTAL_LOSS)
    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)

    loss_iter = np.zeros(train_iter)

    print('Training Start:',datetime.datetime.now())
    for i in range(train_iter):
        batch_H, batch_H_T, batch_X, batch_Y = data.generate_data(train_batch_size, K_orig, N_orig, snr_low, snr_high, mod, a)
        train_step.run(feed_dict={H: batch_H, Y: batch_Y, H_T: batch_H_T, X: batch_X})

        loss_iter[i] = np.array(sess.run(SER, {H: batch_H, Y: batch_Y, H_T: batch_H_T, X: batch_X}))

        if i % 1000 == 0:
            batch_H, batch_H_T, batch_X, batch_Y = data.generate_data(train_batch_size, K_orig, N_orig, snr_low, snr_high, mod, a)

            results = sess.run([TOTAL_LOSS,SER], {H: batch_H, Y: batch_Y, H_T: batch_H_T, X: batch_X})
            print_string = [i] + results
            print('%.6d \t Loss: %3.6f \t SER: %1.6f \t\t' %(i, results[0], results[1]), datetime.datetime.now() )

            # Save learnable parameters
            learned_parameters = np.array(sess.run([delta, theta, alpha, beta, gamma], {H: batch_H, Y: batch_Y, H_T: batch_H_T, X: batch_X}))
            np.save('trained_parameters'+'/ADMM_Net_parameters_('+str(K_orig)+'x'+str(N_orig)+ ' ' + str(mod)
                        +'QAM MIMO)', learned_parameters)


    print('Training Finish:',datetime.datetime.now())


# Testing the MIMO detector
print('Test Start:',datetime.datetime.now())
snrdb_list = np.linspace(snrdb_low, snrdb_high, num_snr)
snr_list = 10.0 ** (snrdb_list / 10.0)
for j in range(num_snr):
    for jj in range(test_iter):
        batch_H, batch_H_T, batch_X, batch_Y = data.generate_data(test_batch_size, K_orig, N_orig, snr_list[j], snr_list[j], mod, a)

        if detector_name =="ZF":
            results = sess.run([SER],{H: batch_H, Y: batch_Y, H_T: batch_H_T, X: batch_X})

        else:
            results = sess.run([SER, layer_ser],{H: batch_H, Y: batch_Y, H_T: batch_H_T, X: batch_X})
            layer_ser_mean[:,j] = layer_ser_mean[:,j] + results[1]

        tic = tm.time()
        tmp_ser_iter[:, jj] = np.array(sess.run(SER,{H: batch_H, Y: batch_Y, H_T: batch_H_T, X: batch_X,}))
        toc = tm.time()
        tmp_times[0][jj] = toc - tic
        tmp_sers[0][jj] = results[0]

        if jj % 10 == 0:
            print('snr:',snrdb_list[j],' %8d   /%8d\t\t' %(int((jj+10)*test_batch_size), int(test_iter*test_batch_size)), datetime.datetime.now())

    sers[0][j] = np.mean(tmp_sers[0])
    times[0][j] = np.mean(tmp_times[0]) / test_batch_size

    if detector_name != "ZF":
        ser_layer = layer_ser_mean / test_iter

print('Test Finish:',datetime.datetime.now())
print('Test result')
print('snrdb_list : ',snrdb_list)
print('sers : ',sers)
print('times : ',times)
