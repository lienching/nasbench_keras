# NAS-Bench 101 to Tensorflow 2.0 (tf.keras) converter
[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://github.com/evgps/nasbench_keras)

The small but powerful tool that converts models from **NAS-Bench 101** [code](https://github.com/google-research/nasbench) [paper](https://arxiv.org/abs/1902.09635) into real tf.keras models
- **Completely standalone (check requirements). Don't need NAS-Bench to be installed!**
- **Require Tensorflow 2.0 or higher**

Can be used for: 
  - Training
  - Latency measurements
  - Latency measurements on devices with TFLite Converter
  - ...


### Installation
Use PyPi

```sh
$ pip3 install nasbench_keras --user
```
or 
```sh
$ git clone https://github.com/evgps/nasbench_keras.git
$ cd nasbench_keras
$ pip3 install -e .
```
### Getting Started

To test the tool you can:
  - Download or generate json with all model graphs for original NAS-Bench 101 cells size of 7 or less: [GDrive](https://drive.google.com/open?id=1yClNzQ8DCGW-iYwroA7HWKUrqTeosTev)
  
```python
import tensorflow as tf
import json
from nasbench_keras import ModelSpec, build_keras_model, build_module

with open('generated_graphs.json', "rb") as f:
    models = json.load(f)

# Get model by the hash
model = models['0001a2f6c8977346ccd12fa0c435bf42']

# Adjacency matrix and nuberically-coded layer list
matrix, labels = model

# Configure whole network
config = {'available_ops' : ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'],
        'stem_filter_size' : 128,
        'data_format' : 'channels_first',
        'num_stacks' : 3,
        'num_modules_per_stack' : 2,
        'num_labels' : 1000}

# Transfer numerically-coded operations to layers (check base_ops.py)
labels = (['input'] + [config['available_ops'][l] for l in labels[1:-1]] + ['output'])

# Module graph
spec = ModelSpec(matrix, labels, data_format='channels_first')

# Create module
inputs = tf.keras.layers.Input((3,224,224), 1)
outputs = build_module(spec=spec, inputs=inputs, channels=128, is_training=True)
module = tf.keras.Model(inputs=inputs, outputs=outputs)
module.summary()

# Create whole network with same config
features = tf.keras.layers.Input((3,224,224), 1)
net_outputs = build_keras_model(spec, features, labels, config)
net = tf.keras.Model(inputs=features, outputs=net_outputs)
net.summary()
```

