# coding=utf-8
import numpy as np
import struct
import os
import time

class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride, type=0):
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        self.forward = self.forward_raw
        self.backward = self.backward_raw
        # TODO: type 修改回来
        if type == 1:  # type 设为 1 时，使用优化后的 foward 和 backward 函数
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup
        print('\tConvolutional layer with kernel size %d, input channel %d, output channel %d.' % (self.kernel_size, self.channel_in, self.channel_out))
    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out))
        self.bias = np.zeros([self.channel_out])
    def forward_raw(self, input):
        print('conv forward raw')
        start_time = time.time()
        self.input = input # [N, C, H, W]
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]] = self.input
        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (width - self.kernel_size) // self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.channel_out, height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.channel_out):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO: 计算卷积层的前向传播，特征图与卷积核的内积再加偏置
                        self.output[idxn, idxc, idxh, idxw] = np.sum(self.input_pad[idxn, :, idxh * self.stride : idxh * self.stride + self.kernel_size, idxw * self.stride : idxw * self.stride + self.kernel_size] * self.weight[:, :, :, idxc]) + self.bias[idxc]
        self.forward_time = time.time() - start_time
        return self.output
    def forward_speedup(self, input):
        print('conv forward speedup')
        # TODO: 改进forward函数，使得计算加速
        start_time = time.time()

        S_N = input.shape[0]
        S_Cin = self.channel_in
        S_Cout = self.channel_out
        S_Ker = self.kernel_size
        S_Hin = input.shape[2]
        S_Win = input.shape[3]
        S_Hpin = S_Hin + self.padding * 2
        S_Wpin = S_Win + self.padding * 2
        S_Hout = (S_Hpin - S_Ker) // self.stride + 1
        S_Wout = (S_Wpin - S_Ker) // self.stride + 1
        # [N, C, H, W]
        self.input = input 
        # [N, Cin, Hpin, Wpin]
        self.input_pad = np.zeros([S_N, S_Cin, S_Hpin, S_Wpin])
        self.input_pad[:, :, self.padding : self.padding + S_Hin, self.padding : self.padding + S_Win] = self.input
        # 卷积核向量化 [Cin, K, K, Cout] => [Cin * K * K, Cout]
        self.weight_reshape = np.reshape(self.weight, [-1, self.channel_out])
        # 输入向量化 [N, Hout * Wout, Cin * K * K]
        self.img2col = np.zeros([S_N, S_Hout * S_Wout, S_Cin * S_Ker * S_Ker])
        for idxh in range(S_Hout):
            for idxw in range(S_Wout):
                # [N, Cin, K, K] => [N, 1, Cin * K * K]
                self.img2col[:, idxh * S_Wout + idxw, :] = self.input_pad \
                [:, :, idxh * self.stride : idxh * self.stride + S_Ker, idxw * self.stride : idxw * self.stride + S_Ker].reshape([S_N, 1, S_Cin * S_Ker * S_Ker])
        # [N, Hout * Wout, Cin * K * K] * [Cin * K * K, Cout] => [N, Hout * Wout, Cout]
        # [N, Hout * Wout, Cout] => [N * Hout * Wout, Cout] + [Cout]
        output = np.matmul(self.img2col, self.weight_reshape).reshape([S_N * S_Hout * S_Wout, S_Cout]) + self.bias
        # [N * Hout * Wout, Cout] => [N, Hout, Wout, Cout] => [N, Cout, Hout, Wout]
        self.output = output.reshape([S_N, S_Hout, S_Wout, S_Cout]).transpose([0, 3, 1, 2]) 

        self.forward_time = time.time() - start_time
        print('conv forward speedup time: %f ms'%(self.forward_time*1000))
        return self.output
    def backward_speedup(self, top_diff):
        print('conv backward speedup')
        # TODO: 改进backward函数，使得计算加速
        start_time = time.time()

        S_N = self.input.shape[0]
        S_Cin = self.channel_in
        S_Cout = self.channel_out
        S_Ker = self.kernel_size
        S_Hpin = self.input_pad.shape[2]
        S_Wpin = self.input_pad.shape[3]
        S_Hout = top_diff.shape[2]
        S_Wout = top_diff.shape[3]
        
        # [N, Cout, Hout, Wout] => [N, Hout, Wout, Cout] => [N * Hout * Wout, Cout]
        top_diff_col = np.transpose(top_diff, [0, 2, 3, 1]).reshape([-1, S_Cout]) 
        """
        Calculate d_weight and d_bias
        """
        # [Cin, K * K, N * Hout * Wout]
        ##input_pad_col = np.zeros([S_Cin, S_Ker * S_Ker, S_N * S_Hout * S_Wout]) 
        # input_pad [N, Cin, Hpin, Wpin]
        ##for idxh in range(S_Ker):     # K
            ##for idxw in range(S_Ker): # K
                # [N, Cin, Hout, Wout] => [Cin, N, Hout, Wout] => [Cin, 1, N * Hout * Wout]
                ##input_pad_col[:, idxh * S_Ker + idxw, :] = np.transpose(self.input_pad[:, :, idxh : idxh + S_Hout * self.stride, idxw : idxw + S_Wout * self.stride], [1, 0, 2, 3]).reshape([S_Cin, S_N * S_Hout * S_Wout])
        # [Cin, K * K, N * Hout * Wout] * [N * Hout * Wout, Cout] => [Cin, K, K, Cout]
        ##self.d_weight = np.dot(input_pad_col, top_diff_col).reshape(self.weight.shape)
        # [N * Hout * Wout, Cout] => [1 * Cout] 
        ##self.d_bias = np.sum(top_diff_col, axis=0).reshape(self.bias.shape) 
        """
        Calculate bottom_diff
        """
        # [N * Hout * Wout, Cout] * [Cout, Cin * K * K] => [N * Hout * Wout, Cin * K * K] => [N, Hout * Wout, Cin * K * K]
        bottom_diff_col = np.dot(top_diff_col, self.weight_reshape.T).reshape([top_diff.shape[0], top_diff.shape[2] * top_diff.shape[3], -1])
        # [N, Hout * Wout, Cin * K * K] => [N, Cin * K * K, Hout * Wout]
        bottom_diff_col = np.transpose(bottom_diff_col, [0, 2, 1])
        # [N, Cin, Hpin, Wpin]
        bottom_diff_pad = np.zeros([S_N, S_Cin, S_Hpin, S_Wpin])
        for idxh in range(S_Hout):
            for idxw in range(S_Wout):
                # [N, Cin * K * K, 1] => [N, Cin, K, K]
                bottom_diff_pad[:, :, idxh * self.stride : idxh * self.stride + S_Ker, idxw * self.stride : idxw * self.stride + S_Ker] += bottom_diff_col[:, :, idxh * S_Wout + idxw].reshape([S_N, S_Cin, S_Ker, S_Ker])
        # [N, Cin, Hpin, Wpin] => [N, Cin, Hin, Win]
        bottom_diff = bottom_diff_pad[:, :, self.padding : S_Hpin - self.padding, self.padding : S_Wpin - self.padding]

        self.backward_time = time.time() - start_time
        print('conv forward speedup time: %f ms'%(self.backward_time*1000))
        return bottom_diff
    def backward_raw(self, top_diff):
        start_time = time.time()
        self.d_weight = np.zeros(self.weight.shape)
        self.d_bias = np.zeros(self.bias.shape)
        bottom_diff = np.zeros(self.input_pad.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        # TODO： 计算卷积层的反向传播， 权重、偏置的梯度和本层损失
                        self.d_weight[:, :, :, idxc] += top_diff[idxn, idxc, idxh, idxw] * self.input_pad[idxn, :, idxh * self.stride : idxh * self.stride + self.kernel_size, idxw * self.stride : idxw * self.stride + self.kernel_size]
                        self.d_bias[idxc] += top_diff[idxn, idxc, idxh, idxw]
                        bottom_diff[idxn, :, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size] += top_diff[idxn, idxc, idxh, idxw] * self.weight[:, : self.kernel_size, : self.kernel_size, idxc]
        bottom_diff = bottom_diff[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]]  
        self.backward_time = time.time() - start_time
        return bottom_diff
    def get_gradient(self):
        return self.d_weight, self.d_bias
    def update_param(self, lr):
        self.weight += - lr * self.d_weight
        self.bias += - lr * self.d_bias
    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
    def get_forward_time(self):
        return self.forward_time
    def get_backward_time(self):
        return self.backward_time

class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride, type=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.forward = self.forward_raw
        self.backward = self.backward_raw_book
        # TODO: type 修改回来
        if type == 1:  # type 设为 1 时，使用优化后的 foward 和 backward 函数
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup
        print('\tMax pooling layer with kernel size %d, stride %d.' % (self.kernel_size, self.stride))
    def forward_raw(self, input):
        print('Max pooling layer forward raw.')
        start_time = time.time()
        self.input = input # [N, C, H, W]
        self.max_index = np.zeros(self.input.shape)
        height_out = (self.input.shape[2] - self.kernel_size) // self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size) // self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.input.shape[1], height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO： 计算最大池化层的前向传播， 取池化窗口内的最大值
                        self.output[idxn, idxc, idxh, idxw] = np.max(self.input[idxn, idxc, idxh * self.stride : idxh * self.stride + self.kernel_size, idxw * self.stride : idxw  * self.stride + self.kernel_size])
                        curren_max_index = np.argmax(self.input[idxn, idxc, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size])
                        curren_max_index = np.unravel_index(curren_max_index, [self.kernel_size, self.kernel_size])
                        self.max_index[idxn, idxc, idxh*self.stride+curren_max_index[0], idxw*self.stride+curren_max_index[1]] += 1
        return self.output
    def forward_speedup(self, input):
        print('Max pooling layer forward speedup.')
        # TODO: 改进forward函数，使得计算加速
        start_time = time.time()

        S_N = input.shape[0]
        S_C = input.shape[1]
        S_Ker = self.kernel_size
        S_Hint = input.shape[2]
        S_Wint = input.shape[3]
        S_Hout = (S_Hint - self.kernel_size) // self.stride + 1
        S_Wout = (S_Wint - self.kernel_size) // self.stride + 1

        # [N, C, Hin, Win]
        self.input = input 
        # [N, C, Hout, Wout, K * K]
        self.input_col = np.zeros([S_N, S_C, S_Hout, S_Wout, S_Ker * S_Ker])
        for idxh in range(S_Hout):
            for idxw in range(S_Wout):
                # [N, C, K, K] => [N, C, 1, 1, K * K]
                self.input_col[:, :, idxh, idxw, :] = self.input \
                [:, :, idxh * self.stride : idxh * self.stride + S_Ker, idxw * self.stride : idxw * self.stride + S_Ker].reshape([S_N, S_C, S_Ker * S_Ker])
        # [N, C, Hout, Wout, K * K] => [N, C, Hout, Wout, 1]
        self.output = np.max(self.input_col, axis=4, keepdims=True)
        # [N, C, Hout, Wout, K * K] value is True or False
        self.max_index = (self.input_col == self.output)
        # [N, C, Hout, Wout, 1] => [N, C, Hout, Wout]
        self.output = self.output.reshape([S_N, S_C, S_Hout, S_Wout])
        
        return self.output
    def backward_speedup(self, top_diff):
        print('Max pooling layer backward speedup.')
        # TODO: 改进backward函数，使得计算加速
        
        S_N = top_diff.shape[0]
        S_C = top_diff.shape[1]
        S_Ker = self.kernel_size
        S_Hout = top_diff.shape[2]
        S_Wout = top_diff.shape[3]

        # [N, C, Hout, Wout, K * K] * [N, C, Hout, Wout, 1] => [N, C, Hout, Wout, K * K] * [N, C, Hout, Wout, 1]
        pool_diff = (self.max_index * top_diff[:, :, :, :, np.newaxis])
        # [N, C, Hin, Win]
        bottom_diff = np.zeros(self.input.shape)
        for idxh in range(S_Hout):
            for idxw in range(S_Wout):
                # [N, C, 1, 1, K * K] => [N, C, K, K]
                bottom_diff[:, :, idxh * self.stride : idxh * self.stride + S_Ker, idxw * self.stride : idxw * self.stride + S_Ker] += pool_diff[:, :, idxh, idxw, :].reshape([S_N, S_C, self.kernel_size, self.kernel_size])

        return bottom_diff
    def backward_raw_book(self, top_diff):
        bottom_diff = np.zeros(self.input.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        # TODO: 最大池化层的反向传播， 计算池化窗口中最大值位置， 并传递损失
                        max_index = np.argwhere(self.max_index[idxn, idxc, \
                            idxh * self.stride : idxh * self.stride + self.kernel_size, \
                            idxw * self.stride : idxw * self.stride + self.kernel_size] > 0)[0]
                        bottom_diff[idxn, idxc, idxh*self.stride+max_index[0], idxw*self.stride+max_index[1]] = top_diff[idxn, idxc, idxh, idxw]
                        self.max_index[idxn, idxc, idxh*self.stride+max_index[0], idxw*self.stride+max_index[1]] -= 1
        return bottom_diff

class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert np.prod(self.input_shape) == np.prod(self.output_shape)
        print('\tFlatten layer with input shape %s, output shape %s.' % (str(self.input_shape), str(self.output_shape)))
    def forward(self, input):
        assert list(input.shape[1:]) == list(self.input_shape)
        # matconvnet feature map dim: [N, height, width, channel]
        # ours feature map dim: [N, channel, height, width]
        self.input = np.transpose(input, [0, 2, 3, 1])
        self.output = self.input.reshape([self.input.shape[0]] + list(self.output_shape))
        return self.output
    def backward(self, top_diff):
        assert list(top_diff.shape[1:]) == list(self.output_shape)
        top_diff = np.transpose(top_diff, [0, 3, 1, 2])
        bottom_diff = top_diff.reshape([top_diff.shape[0]] + list(self.input_shape))
        return bottom_diff
