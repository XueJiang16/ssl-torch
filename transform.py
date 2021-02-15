import numpy as np
import torch
from scipy import signal

import math
import cv2

import random
class Transform:
    def __init__(self):
        pass




    def add_noise(self, signal, noise_amount):
        """
        adding noise
        """
        signal = signal.T
        noise = (0.4 ** 0.5) * np.random.normal(1, noise_amount, np.shape(signal)[0])
        noise = noise[:,None]
        noised_signal = signal + noise
        noised_signal = noised_signal.T
        # print(noised_signal.shape)
        return noised_signal

    def add_noise_with_SNR(self,signal, noise_amount):
        """
        adding noise
        created using: https://stackoverflow.com/a/53688043/10700812
        """
        signal = signal[0]
        target_snr_db = noise_amount  # 20
        x_watts = signal ** 2  # Calculate signal power and convert to dB
        sig_avg_watts = np.mean(x_watts)
        sig_avg_db = 10 * np.log10(sig_avg_watts)  # Calculate noise then convert to watts
        noise_avg_db = sig_avg_db - target_snr_db
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        mean_noise = 0
        noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts),
                                       len(x_watts))  # Generate an sample of white noise
        noised_signal = signal + noise_volts  # noise added signal
        noised_signal = noised_signal[None,:]
        # print(noised_signal.shape)

        return noised_signal

    def scaled(self,signal, factor_list):
        """"
        scale the signal
        """
        factor = round(np.random.uniform(factor_list[0],factor_list[1]),2)
        signal[0] = 1 / (1 + np.exp(-signal[0]))
        # print(signal.max())
        return signal

    def negate(self,signal):
        """
        negate the signal
        """
        signal[0] = signal[0] * (-1)
        return signal

    def hor_filp(self,signal):
        """
        flipped horizontally
        """
        hor_flipped = np.flip(signal,axis=1)
        return hor_flipped

    def permute(self,signal, pieces):
        """
        signal: numpy array (batch x window)
        pieces: number of segments along time
        """
        signal = signal.T
        pieces = int(np.ceil(np.shape(signal)[0] / (np.shape(signal)[0] // pieces)).tolist()) #向上取整
        piece_length = int(np.shape(signal)[0] // pieces)

        sequence = list(range(0, pieces))
        np.random.shuffle(sequence)

        permuted_signal = np.reshape(signal[:(np.shape(signal)[0] // pieces * pieces)],
                                     (pieces, piece_length)).tolist()

        tail = signal[(np.shape(signal)[0] // pieces * pieces):]
        permuted_signal = np.asarray(permuted_signal)[sequence]
        permuted_signal = np.concatenate(permuted_signal, axis=0)
        permuted_signal = np.concatenate((permuted_signal,tail[:,0]), axis=0)
        permuted_signal = permuted_signal[:,None]
        permuted_signal = permuted_signal.T
        return permuted_signal

    def cutout_resize(self,signal,pieces):
        """
                signal: numpy array (batch x window)
                pieces: number of segments along time
                cutout 1 piece
                """
        signal = signal.T
        pieces = int(np.ceil(np.shape(signal)[0] / (np.shape(signal)[0] // pieces)).tolist())  # 向上取整
        piece_length = int(np.shape(signal)[0] // pieces)
        import random
        sequence = []

        cutout = random.randint(0, pieces)
        # print(cutout)
        # sequence1 = list(range(0, cutout))
        # sequence2 = list(range(int(cutout + 1), pieces))
        # sequence = np.hstack((sequence1, sequence2))
        for i in range(pieces):
            if i == cutout:
                pass
            else:
                sequence.append(i)
        # print(sequence)

        cutout_signal = np.reshape(signal[:(np.shape(signal)[0] // pieces * pieces)],
                                     (pieces, piece_length)).tolist()

        tail = signal[(np.shape(signal)[0] // pieces * pieces):]

        cutout_signal = np.asarray(cutout_signal)[sequence]

        cutout_signal = np.hstack(cutout_signal)
        cutout_signal = np.concatenate((cutout_signal, tail[:, 0]), axis=0)

        cutout_signal = cv2.resize(cutout_signal, (1, 3072), interpolation=cv2.INTER_LINEAR)
        cutout_signal = cutout_signal.T


        return cutout_signal

    def cutout_zero(self,signal,pieces):
        """
                signal: numpy array (batch x window)
                pieces: number of segments along time
                cutout 1 piece
                """
        signal = signal.T
        ones = np.ones((np.shape(signal)[0],np.shape(signal)[1]))
        # print(ones.shape)
        # assert False
        pieces = int(np.ceil(np.shape(signal)[0] / (np.shape(signal)[0] // pieces)).tolist())  # 向上取整
        piece_length = int(np.shape(signal)[0] // pieces)


        cutout = random.randint(1, pieces)
        cutout_signal = np.reshape(signal[:(np.shape(signal)[0] // pieces * pieces)],
                                     (pieces, piece_length)).tolist()
        ones_pieces = np.reshape(ones[:(np.shape(signal)[0] // pieces * pieces)],
                                   (pieces, piece_length)).tolist()
        tail = signal[(np.shape(signal)[0] // pieces * pieces):]

        cutout_signal = np.asarray(cutout_signal)
        ones_pieces = np.asarray(ones_pieces)
        for i in range(pieces):
            if i == cutout:
                ones_pieces[i]*=0

        cutout_signal = cutout_signal * ones_pieces
        cutout_signal = np.hstack(cutout_signal)
        cutout_signal = np.concatenate((cutout_signal, tail[:, 0]), axis=0)
        cutout_signal = cutout_signal[:,None]
        cutout_signal = cutout_signal.T

        return cutout_signal
    # mic
    def crop_resize(self, signal, size):
        signal = signal.T
        size = signal.shape[0] * size
        size = int(size)
        start = random.randint(0, signal.shape[0]-size)
        crop_signal = signal[start:start + size,:]
        # print(crop_signal.shape)

        crop_signal = cv2.resize(crop_signal, (1, 3072), interpolation=cv2.INTER_LINEAR)
        # print(crop_signal.shape)
        crop_signal = crop_signal.T
        return crop_signal

    def move_avg(self,a,n, mode="same"):
        # a = a.T

        result = np.convolve(a[0], np.ones((n,)) / n, mode=mode)
        return result[None,:]

    def bandpass_filter(self, x, order, cutoff, fs=100):
        result = np.zeros((x.shape[0], x.shape[1]))
        w1 = 2 * cutoff[0] / int(fs)
        w2 = 2 * cutoff[1] / int(fs)
        b, a = signal.butter(order, [w1, w2], btype='bandpass')  # 配置滤波器 8 表示滤波器的阶数
        result = signal.filtfilt(b, a, x, axis=1)
        # print(result.shape)

        return result

    def lowpass_filter(self, x, order, cutoff, fs=100):
        result = np.zeros((x.shape[0], x.shape[1]))
        w1 = 2 * cutoff[0] / int(fs)
        # w2 = 2 * cutoff[1] / fs
        b, a = signal.butter(order, w1, btype='lowpass')  # 配置滤波器 8 表示滤波器的阶数
        result = signal.filtfilt(b, a, x, axis=1)
        # print(result.shape)

        return result

    def highpass_filter(self, x, order, cutoff, fs=100):
        result = np.zeros((x.shape[0], x.shape[1]))
        w1 = 2 * cutoff[0] / int(fs)
        # w2 = 2 * cutoff[1] / fs
        b, a = signal.butter(order, w1, btype='highpass')  # 配置滤波器 8 表示滤波器的阶数
        result = signal.filtfilt(b, a, x, axis=1)
        # print(result.shape)

        return result


    def time_warp(self,signal, sampling_freq, pieces, stretch_factor, squeeze_factor):
        """
        signal: numpy array (batch x window)
        sampling freq
        pieces: number of segments along time
        stretch factor
        squeeze factor
        """
        signal = signal.T

        total_time = np.shape(signal)[0] // sampling_freq
        segment_time = total_time / pieces
        sequence = list(range(0, pieces))
        stretch = np.random.choice(sequence, math.ceil(len(sequence) / 2), replace=False)
        squeeze = list(set(sequence).difference(set(stretch)))
        initialize = True
        for i in sequence:
            orig_signal = signal[int(i * np.floor(segment_time * sampling_freq)):int(
                (i + 1) * np.floor(segment_time * sampling_freq))]
            orig_signal = orig_signal.reshape(np.shape(orig_signal)[0], 1)
            if i in stretch:
                output_shape = int(np.ceil(np.shape(orig_signal)[0] * stretch_factor))
                new_signal = cv2.resize(orig_signal, (1, output_shape), interpolation=cv2.INTER_LINEAR)
                if initialize == True:
                    time_warped = new_signal
                    initialize = False
                else:
                    time_warped = np.vstack((time_warped, new_signal))
            elif i in squeeze:
                output_shape = int(np.ceil(np.shape(orig_signal)[0] * squeeze_factor))
                new_signal = cv2.resize(orig_signal, (1, output_shape), interpolation=cv2.INTER_LINEAR)
                if initialize == True:
                    time_warped = new_signal
                    initialize = False
                else:
                    time_warped = np.vstack((time_warped, new_signal))
        time_warped = cv2.resize(time_warped, (1,3072), interpolation=cv2.INTER_LINEAR)
        time_warped = time_warped.T
        return time_warped

if __name__ == '__main__':
    from transform import Transform
    import matplotlib.pyplot as plt
    Trans = Transform()
    input = np.zeros((1,3072))
    input = Trans.add_noise(input,10)
    plt.subplot(211)
    plt.plot(input[0])

    # print(input.shape)
    # output = Trans.cutout_resize(input,10)
    order = random.randint(3, 10)
    cutoff = random.uniform(5, 20)
    output = Trans.filter(input, order, [2,15], mode='lowpass')
    plt.subplot(212)
    plt.plot(output[0])
    plt.savefig('filter.png')
    # print(output.shape)
