import kfac
import tensorflow as tf

from models.official.resnet.cifar10_main import _NUM_CLASSES
from models.official.resnet.resnet_model import cifar10_resnet_v2_generator


class kfac_layer_collection:
    def __init__(self):
        import unittest.mock as mock
        self._layer_collection = kfac.LayerCollection()

        def custom_apply(layer, inputs, *args, **kwargs):
            outs = layer.__call__(inputs, *args, **kwargs)
            if isinstance(layer, tf.layers.Conv2D):
                variables = tuple(layer.trainable_variables) if len(layer.trainable_variables) > 1 else \
                    layer.trainable_variables[0]
                self._layer_collection.register_conv2d(variables, [1] + list(layer.strides) + [1],
                                                       layer.padding.upper(), inputs, outs)
            elif isinstance(layer, tf.layers.Dense):
                variables = tuple(layer.trainable_variables) if len(layer.trainable_variables) > 1 else \
                    layer.trainable_variables[0]
                self._layer_collection.register_fully_connected(variables, inputs, outs)
                self.logit = outs
            elif isinstance(layer, tf.layers.BatchNormalization):
                self._layer_collection.register_generic(tuple(layer.trainable_variables), tf.shape(outs)[0], "diagonal")
            else:
                print("ignored layers for kfac", layer)
                assert len(layer.trainable_variables) == 0
            return outs

        self._patch = mock.patch.object(tf.layers.Layer, "apply", new=custom_apply)

    def __enter__(self):
        self._patch.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert exc_type is None
        if hasattr(self, 'logit'):
            self._layer_collection.register_categorical_predictive_distribution(self.logit)
        else:
            raise ValueError('no logit!')
    @property
    def layer_collection(self):
        return self._layer_collection


def convnet(input, *arg):
    images = input
    with tf.variable_scope('conv1') as scope:
        conv1 = tf.layers.conv2d(images, 64,
                                 kernel_size=5,
                                 strides=1,
                                 padding='SAME',
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(stddev=5e-2))
    # pool1
    pool1 = tf.layers.max_pooling2d(conv1, 2, strides=2, padding='SAME', name='pool1')
    # conv2
    with tf.variable_scope('conv2') as scope:
        conv2 = tf.layers.conv2d(pool1, 64,
                                 kernel_size=5,
                                 strides=1,
                                 padding='SAME',
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(stddev=5e-2))

    # pool2
    pool2 = tf.layers.max_pooling2d(conv2, 2, strides=2, padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        flatten1 = tf.layers.flatten(pool2)
        local3 = tf.layers.dense(flatten1, 384,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.04),
                                 bias_initializer=tf.constant_initializer(0.1))

    # local4
    with tf.variable_scope('local4') as scope:
        local4 = tf.layers.dense(local3, 192,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.04),
                                 bias_initializer=tf.constant_initializer(0.1))

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        softmax_linear = tf.layers.dense(local4, _NUM_CLASSES,
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(stddev=1 / 192.0),
                                         bias_initializer=tf.constant_initializer(0.0))

    return softmax_linear


def mlp(input, *arg):
    input = tf.layers.flatten(input)
    init = tf.random_normal_initializer(mean=0, stddev=0.01)
    out = tf.layers.dense(input, 20, activation=tf.nn.sigmoid, kernel_initializer=init, bias_initializer=init)
    out = tf.layers.dense(out, 10, activation=None, kernel_initializer=init, bias_initializer=init)
    return out


def get_problem(params):
    if params['problem'] == 'resnet':
        network = cifar10_resnet_v2_generator(
            params['resnet_size'], _NUM_CLASSES, params['data_format'])
    elif params['problem'] == 'convnet':
        network = convnet
    elif params['problem'] == 'MLP':
        network = mlp
    else:
        raise ValueError('{} not found!'.format(params['problem']))
    return network
