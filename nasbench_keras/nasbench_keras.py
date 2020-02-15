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

from .model_spec import ModelSpec
import numpy as np
import tensorflow as tf
import json
from .base_ops import conv_bn_relu, Identity, Conv3x3BnRelu, Conv1x1BnRelu, MaxPool3x3, \
                     BottleneckConv3x3, BottleneckConv5x5, MaxPool3x3Conv1x1, OP_MAP

def projection(inputs, channels, is_training, data_format):
    """1x1 projection (as in ResNet) followed by batch normalization and ReLU."""
    net = conv_bn_relu(inputs, 1, channels, is_training, data_format)

    return net

def truncate(inputs, channels, data_format):
    """Slice the inputs to channels if necessary."""
    if data_format == 'channels_last':
        input_channels = inputs.get_shape()[3]
    else:
        assert data_format == 'channels_first'
        input_channels = inputs.get_shape()[1]

    if input_channels < channels:
        raise ValueError('input channel < output channels for truncate')
    elif input_channels == channels:
        return inputs   # No truncation necessary
    else:
        # Truncation should only be necessary when channel division leads to
        # vertices with +1 channels. The input vertex should always be projected to
        # the minimum channel count.
        assert input_channels - channels == 1
        if data_format == 'channels_last':
            return tf.slice(inputs, [0, 0, 0, 0], [-1, -1, -1, channels])
        else:
            return tf.slice(inputs, [0, 0, 0, 0], [-1, channels, -1, -1])


def compute_vertex_channels(input_channels, output_channels, matrix):
    """Computes the number of channels at every vertex.
    Given the input channels and output channels, this calculates the number of
    channels at each interior vertex. Interior vertices have the same number of
    channels as the max of the channels of the vertices it feeds into. The output
    channels are divided amongst the vertices that are directly connected to it.
    When the division is not even, some vertices may receive an extra channel to
    compensate.
    Args:
        input_channels: input channel count.
        output_channels: output channel count.
        matrix: adjacency matrix for the module (pruned by model_spec).
    Returns:
        list of channel counts, in order of the vertices.
    """
    num_vertices = np.shape(matrix)[0]

    vertex_channels = [0] * num_vertices
    vertex_channels[0] = input_channels
    vertex_channels[num_vertices - 1] = output_channels

    if num_vertices == 2:
        # Edge case where module only has input and output vertices
        return vertex_channels

    # Compute the in-degree ignoring input, axis 0 is the src vertex and axis 1 is
    # the dst vertex. Summing over 0 gives the in-degree count of each vertex.
    in_degree = np.sum(matrix[1:], axis=0)
    interior_channels = output_channels // in_degree[num_vertices - 1]
    correction = output_channels % in_degree[num_vertices - 1]  # Remainder to add

    # Set channels of vertices that flow directly to output
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            vertex_channels[v] = interior_channels
            if correction:
                vertex_channels[v] += 1
                correction -= 1

    # Set channels for all other vertices to the max of the out edges, going
    # backwards. (num_vertices - 2) index skipped because it only connects to
    # output.
    for v in range(num_vertices - 3, 0, -1):
        if not matrix[v, num_vertices - 1]:
            for dst in range(v + 1, num_vertices - 1):
                if matrix[v, dst]:
                    vertex_channels[v] = max(vertex_channels[v], vertex_channels[dst])
        assert vertex_channels[v] > 0

    # Sanity check, verify that channels never increase and final channels add up.
    final_fan_in = 0
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            final_fan_in += vertex_channels[v]
        for dst in range(v + 1, num_vertices - 1):
            if matrix[v, dst]:
                assert vertex_channels[v] >= vertex_channels[dst]
    assert final_fan_in == output_channels or num_vertices == 2
    # num_vertices == 2 means only input/output nodes, so 0 fan-in

    return vertex_channels


def _covariance_matrix(activations):
    """Computes the unbiased covariance matrix of the samples within the batch.
    Computes the sample covariance between the samples in the batch. Specifically,
        C(i,j) = (x_i - mean(x_i)) dot (x_j - mean(x_j)) / (N - 1)
    Matches the default behavior of np.cov().
    Args:
        activations: tensor activations with batch dimension first.
    Returns:
        [batch, batch] shape tensor for the covariance matrix.
    """
    batch_size = activations.get_shape()[0].value
    flattened = tf.reshape(activations, [batch_size, -1])
    means = tf.reduce_mean(flattened, axis=1, keepdims=True)

    centered = flattened - means
    squared = tf.matmul(centered, tf.transpose(centered))
    cov = squared / (tf.cast(tf.shape(flattened)[1], tf.float32) - 1)

    return cov


def build_module(spec, inputs, channels, is_training):
    """Build a custom module using a proposed model spec.

    Builds the model using the adjacency matrix and op labels specified. Channels
    controls the module output channel count but the interior channels are
    determined via equally splitting the channel count whenever there is a
    concatenation of Tensors.

    Args:
        spec: ModelSpec object.
        inputs: input Tensors to this module.
        channels: output channel count.
        is_training: bool for whether this model is training.

    Returns:
        output Tensor from built module.

    Raises:
        ValueError: invalid spec
    """
    num_vertices = np.shape(spec.matrix)[0]

    if spec.data_format == 'channels_last':
        channel_axis = 3
    elif spec.data_format == 'channels_first':
        channel_axis = 1
    else:
        raise ValueError('invalid data_format')

    input_channels = inputs.get_shape()[channel_axis]
    # vertex_channels[i] = number of output channels of vertex i
    vertex_channels = compute_vertex_channels(
            input_channels, channels, spec.matrix)

    # Construct tensors from input forward
    tensors = [tf.identity(inputs, name='input')]

    final_concat_in = []
    for t in range(1, num_vertices - 1):
        # Create interior connections, truncating if necessary
        add_in = [truncate(tensors[src], vertex_channels[t], spec.data_format)
                            for src in range(1, t) if spec.matrix[src, t]]

        # Create add connection from projected input
        if spec.matrix[0, t]:
            add_in.append(projection(
                    tensors[0],
                    vertex_channels[t],
                    is_training,
                    spec.data_format))

        if len(add_in) == 1:
            vertex_input = add_in[0]
        else:
            vertex_input = tf.add_n(add_in)

        # Perform op at vertex t
        op = OP_MAP[spec.ops[t]](
                is_training=is_training,
                data_format=spec.data_format)
        vertex_value = op.build(vertex_input, vertex_channels[t])

        tensors.append(vertex_value)
        if spec.matrix[t, num_vertices - 1]:
            final_concat_in.append(tensors[t])

    # Construct final output tensor by concating all fan-in and adding input.
    if not final_concat_in:
        # No interior vertices, input directly connected to output
        assert spec.matrix[0, num_vertices - 1]
        outputs = projection(
                tensors[0],
                channels,
                is_training,
                spec.data_format)

    else:
        if len(final_concat_in) == 1:
            outputs = final_concat_in[0]
        else:
            outputs = tf.concat(final_concat_in, channel_axis)

        if spec.matrix[0, num_vertices - 1]:
            outputs += projection(
                    tensors[0],
                    channels,
                    is_training,
                    spec.data_format)

    outputs = tf.identity(outputs, name='output')
    return outputs


def build_keras_model(spec, features, labels, config):
    """Builds the model from the input features."""
    
    is_training = True

    # Store auxiliary activations increasing in depth of network. First
    # activation occurs immediately after the stem and the others immediately
    # follow each stack.
    aux_activations = []
    if config['data_format'] == 'channels_last':
        channel_axis = 3
    elif config['data_format'] == 'channels_first':
        channel_axis = 1

    # Initial stem convolution
    net = conv_bn_relu(
            features, 3, config['stem_filter_size'],
            is_training, config['data_format'])
    aux_activations.append(net)

    for stack_num in range(config['num_stacks']):
        channels = net.get_shape()[channel_axis]

        # Downsample at start (except first)
        if stack_num > 0:
            net = tf.keras.layers.MaxPool2D(
                    pool_size=(2, 2),
                    strides=(2, 2),
                    padding='same',
                    data_format=config['data_format'])(net)

            # Double output channels each time we downsample
            channels *= 2

        for module_num in range(config['num_modules_per_stack']):
            net = build_module(
                    spec,
                    inputs=net,
                    channels=channels,
                    is_training=is_training)
        aux_activations.append(net)

    # Global average pool
    if config['data_format'] == 'channels_last':
        net = tf.reduce_mean(net, [1, 2])
    elif config['data_format'] == 'channels_first':
        net = tf.reduce_mean(net, [2, 3])
    else:
        raise ValueError('invalid data_format')

    # Fully-connected layer to labels
    logits = tf.keras.layers.Dense(units=config['num_labels'])(net)

    return logits
