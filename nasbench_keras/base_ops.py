# Copyright evgps

# Licensed under the MIT license:

#     http://www.opensource.org/licenses/mit-license.php

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Code created based on https://github.com/google-research/nasbench

import abc

import tensorflow as tf

# Currently, only channels_last is well supported.
VALID_DATA_FORMATS = frozenset(['channels_last', 'channels_first'])
MIN_FILTERS = 8
BN_MOMENTUM = 0.997
BN_EPSILON = 1e-5


def conv_bn_relu(inputs, conv_size, conv_filters, is_training, data_format):
    """Convolution followed by batch norm and ReLU."""
    if data_format == 'channels_last':
        axis = 3
    elif data_format == 'channels_first':
        axis = 1
    else:
        raise ValueError('invalid data_format')

    net = tf.keras.layers.Conv2D(
            filters=conv_filters,
            kernel_size=conv_size,
            strides=(1, 1),
            use_bias=False,
            kernel_initializer="uniform",
            padding='same',
            data_format=data_format)(inputs)

    net = tf.keras.layers.BatchNormalization(
            axis=axis,
            momentum=BN_MOMENTUM,
            epsilon=BN_EPSILON,
            trainable=is_training)(net)

    net = tf.keras.layers.ReLU()(net)

    return net


class BaseOp(object):
    """Abstract base operation class."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, is_training, data_format='channels_last'):
        self.is_training = is_training
        if data_format.lower() not in VALID_DATA_FORMATS:
            raise ValueError('invalid data_format')
        self.data_format = data_format.lower()

    @abc.abstractmethod
    def build(self, inputs, channels):
        """Builds the operation with input tensors and returns an output tensor.

        Args:
            inputs: a 4-D Tensor.
            channels: int number of output channels of operation. The operation may
                choose to ignore this parameter.

        Returns:
            a 4-D Tensor with the same data format.
        """
        pass


class Identity(BaseOp):
    """Identity operation (ignores channels)."""

    def build(self, inputs, channels):
        del channels    # Unused
        return tf.identity(inputs, name='identity')


class Conv3x3BnRelu(BaseOp):
    """3x3 convolution with batch norm and ReLU activation."""

    def build(self, inputs, channels):
        net = conv_bn_relu(
                inputs, 3, channels, self.is_training, self.data_format)
        return net


class Conv1x1BnRelu(BaseOp):
    """1x1 convolution with batch norm and ReLU activation."""

    def build(self, inputs, channels):
        net = conv_bn_relu(
                inputs, 1, channels, self.is_training, self.data_format)
        return net


class MaxPool3x3(BaseOp):
    """3x3 max pool with no subsampling."""

    def build(self, inputs, channels):
        del channels    # Unused
        net = tf.keras.layers.MaxPool2D(
                pool_size=(3, 3),
                strides=(1, 1),
                padding='same',
                data_format=self.data_format)(inputs)
        return net


class BottleneckConv3x3(BaseOp):
    """[1x1(/4)]+3x3+[1x1(*4)] conv. Uses BN + ReLU post-activation."""
    # TODO(chrisying): verify this block can reproduce results of ResNet-50.

    def build(self, inputs, channels):
        net = conv_bn_relu(
                inputs, 1, channels // 4, self.is_training, self.data_format)
        net = conv_bn_relu(
                net, 3, channels // 4, self.is_training, self.data_format)
        net = conv_bn_relu(
                net, 1, channels, self.is_training, self.data_format)

        return net


class BottleneckConv5x5(BaseOp):
    """[1x1(/4)]+5x5+[1x1(*4)] conv. Uses BN + ReLU post-activation."""

    def build(self, inputs, channels):
        net = conv_bn_relu(
                inputs, 1, channels // 4, self.is_training, self.data_format)
        net = conv_bn_relu(
                net, 5, channels // 4, self.is_training, self.data_format)
        net = conv_bn_relu(
                net, 1, channels, self.is_training, self.data_format)

        return net


class MaxPool3x3Conv1x1(BaseOp):
    """3x3 max pool with no subsampling followed by 1x1 for rescaling."""

    def build(self, inputs, channels):
        net = tf.keras.layers.MaxPool2D(
                pool_size=(3, 3),
                strides=(1, 1),
                padding='same',
                data_format=self.data_format)(inputs)

        net = conv_bn_relu(net, 1, channels, self.is_training, self.data_format)

        return net


# Commas should not be used in op names
OP_MAP = {
        'identity': Identity,
        'conv3x3-bn-relu': Conv3x3BnRelu,
        'conv1x1-bn-relu': Conv1x1BnRelu,
        'maxpool3x3': MaxPool3x3,
        'bottleneck3x3': BottleneckConv3x3,
        'bottleneck5x5': BottleneckConv5x5,
        'maxpool3x3-conv1x1': MaxPool3x3Conv1x1,
}
