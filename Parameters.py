### Parameters
# K_orig : the number of Tx antennas
# N_orig : the number of Rx antennas
# a : activity rate
# L : the number of layer
# train_iter = 0    : Test the performance of a trained model in test phase.
# train_iter > 0    : Train the neural networks

import tensorflow as tf
import numpy as np
import function as func

save_file_name = "trained_parameters"
detector_name = "LADMM"       # Types of MIMO detector : ZF, LADMM

mod = 16     # BPSK(0) and QAM(4, 16, 64)

K_orig = 30  # the number of Tx antennas
N_orig = 60  # the number of Rx antennas
a = 0.1      # activity rate

L = 30       # the number of layers

train_iter = 5000  # 修改
train_batch_size = 2000
test_iter = 10
test_batch_size = 1000

snrdb_low = 8.0     # the lower bound of noise db
snrdb_high = 13.0   # the higher bound of noise db
num_snr = int(snrdb_high - snrdb_low + 1) #SNR的个数
snr_low = 10.0 ** (snrdb_low / 10.0)
snr_high = 10.0 ** (snrdb_high / 10.0)

startingLearningRate = 0.0001  # the initial step size of the gradient descent algorithm
decay_factor = 0.97            #衰减因子
decay_step_size = 800          #步长（原1000）

# initialize momentum parameter gamma
init_gamma = np.ones([L, 1])
for i in range (0,L):
    init_gamma[i] = i/(i+5)

# variable for the simulation result
sers = np.zeros((1, num_snr))
times = np.zeros((1, num_snr))
tmp_sers = np.zeros((1, test_iter))
tmp_times = np.zeros((1, test_iter))
tmp_ser_iter = np.zeros([L, test_iter])
layer_ser_mean = np.zeros([L, num_snr]) #每层在各个信噪比下的平均SER