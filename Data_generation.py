### This code is heavily based on Neev Samuel's implementation of a DetNet (https://github.com/neevsamuel)
# B: batch size
# a: user active rate
# H_: 信道矩阵
# H_T: 信道矩阵的转置
# x_: 发送的符号
# y_: 接收到的符号

from Parameters import *

def generate_data(B, K, N, snr_low, snr_high, mod, a):
    
    # Generate BPSK data
    if mod == 0:    
        H_ = np.random.randn(B, N, K)
        H_T = np.zeros([B, K, N])

        x_ = np.sign(np.random.rand(B, K, 1) - 0.5)
        # Determine active users based on activity rate
        active_users = np.random.choice(range(K), size=int(K * a), replace=False)
        x_[np.setdiff1d(range(K), active_users)] = 0  # Set non-active users' signals to zero
        
        y_ = np.zeros([B, N, 1])

        w = np.random.randn(B, N, 1)
        SNR_ = np.zeros([B])

        for i in range(B):
            SNR = np.random.uniform(low=snr_low, high=snr_high)
            H = H_[i, :, :]
            tmp_snr = (H.T.dot(H)).trace() / K
            H_[i, :, :] = H
            y_[i, :, 0] = (H.dot(x_[i, :, 0]) + w[i, :, 0] * np.sqrt(tmp_snr) / np.sqrt(SNR))
            SNR_[i] = SNR
            H_T[i, :, :] = np.transpose(H)

    # Generate QAM data        
    elif mod == 4 or mod == 16 or mod == 64:   
        x_R = np.random.randint(np.log2(mod), size=(B, K, 1))
        x_R = x_R * 2 - (np.log2(mod)-1)

        x_I = np.random.randint(np.log2(mod), size=(B, K, 1))
        x_I = x_I * 2 - (np.log2(mod)-1)

        x_ = np.concatenate((x_R, x_I), axis=1)
        # Determine active users based on activity rate
        active_users = np.random.choice(range(K), size=int(K * a), replace=False)
        x_[np.setdiff1d(range(K), active_users)] = 0  # Set non-active users' signals to zero
        
        y_ = np.zeros([B, 2 * N, 1])

        H_R = np.random.randn(B,N,K)
        H_I = np.random.randn(B,N,K)
        H_  = np.zeros([B,2*N,2*K])
        H_T = np.zeros([B,2*K,2*N])

        w_R = np.random.randn(B,N,1)
        w_I = np.random.randn(B,N,1)
        w   = np.concatenate((w_R , w_I) , axis = 1)

        SNR_ = np.zeros([B])
        for i in range(B):
            SNR = np.random.uniform(low=snr_low, high=snr_high)
            H = np.concatenate((np.concatenate((H_R[i, :, :], -1 * H_I[i, :, :]), axis=1),
                                np.concatenate((H_I[i, :, :], H_R[i, :, :]), axis=1)), axis=0)
            tmp_snr = (H.T.dot(H)).trace() / (2 * K)
            H_[i, :, :] = H
            y_[i, :,0] = H.dot(x_[i, :,0]) + w[i, :,0] * np.sqrt(tmp_snr) / np.sqrt(SNR)
            SNR_[i] = SNR
            H_T[i, :, :] = np.transpose(H)

    else:
        print('Please check the modulation order.')
        

    return H_, H_T, x_, y_

