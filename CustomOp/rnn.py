import tensorflow as tf
import tensorflow.contrib as tfcb

_EPSILON = 1e-6


def slot_name(l, i, rnn_type):
    return rnn_type + '/layer_{}'.format(l) + '/s_{}'.format(i)


def _proprocess(args):
    out = [tf.tanh(v) for v in args]
    return out


def _get_r1_and_x1(optimizer, var):
    slot_names = ['r_1', 'x_1']
    slots = []
    for _slot in slot_names:
        si = optimizer._get_or_make_slot_with_initializer(var,
                                                          tf.zeros_initializer(dtype=var.dtype.base_dtype),
                                                          tf.TensorShape(var.get_shape().as_list()),
                                                          var.dtype.base_dtype,
                                                          _slot,
                                                          optimizer._name + '/' + _slot)
        slots.append(tf.stop_gradient(si))

    return slots


def generate_rnn_optimizer(name, rnn_type, cell_type='lstm', rnn_layers=(4, 4), prefix='metaoptimizer',
                           softplus=False, need_r1_and_x1=False):
    prefix += '/' + rnn_type
    if cell_type == 'lstm':
        cell_class = tfcb.rnn.LSTMBlockCell
    else:
        raise ValueError('rnn type {} no found.'.format(cell_type))

    def _rnn(*args, optimizer=None, var=None):
        assert isinstance(optimizer, tf.train.Optimizer)
        assert isinstance(var, tf.Variable)
        assert hasattr(optimizer, 'set_slot_shadow')
        rnn_list = [cell_class(i) for i in rnn_layers]
        orig_shape = var.get_shape().as_list()
        args = list(args)
        if need_r1_and_x1:
            args += _get_r1_and_x1(optimizer, var)
        args = [tf.reshape(v, [-1]) for v in args]
        out = tf.stack(_proprocess(args), axis=1)

        for l, cell in enumerate(rnn_list):
            state = []
            for i, s in enumerate(cell.state_size):
                si = optimizer._get_or_make_slot_with_initializer(var,
                                                                  tf.zeros_initializer(dtype=var.dtype.base_dtype),
                                                                  tf.TensorShape([out.get_shape().as_list()[0], s]),
                                                                  var.dtype.base_dtype,
                                                                  slot_name(l, i, rnn_type),
                                                                  optimizer._name + '/' + slot_name(l, i, rnn_type))
                state.append(si)
            state = tuple(state)
            with tf.variable_scope(prefix):
                with tf.variable_scope(name + '/layers_{}'.format(l), reuse=tf.AUTO_REUSE):
                    out, state = cell(out, state)
            for i, s in enumerate(state):
                optimizer.set_slot_shadow(var, s, slot_name(l, i, rnn_type))
        with tf.variable_scope(prefix):
            with tf.variable_scope(name + '/dense', reuse=tf.AUTO_REUSE):
                out = tf.layers.dense(out, 1)
        if softplus is True:
            out = tf.nn.softplus(out[:, 0])
        return tf.reshape(out, orig_shape)
    return _rnn


def generate_rnn_state_transform():
    def _rnn(var=None, optimizer=None, **kwargs):
        assert isinstance(optimizer, tf.train.Optimizer)
        assert isinstance(var, tf.Variable)
        assert hasattr(optimizer, 'set_slot_shadow')
        for val_name, val in kwargs.items():
            optimizer.set_slot_shadow(var, val, val_name)
    return _rnn


def generate_x_y_sf(name, x_use='rnn', y_use='rnn'):
    assert x_use in ['rnn', 'x', 'd']
    if x_use == 'rnn':
        x = generate_rnn_optimizer(name, 'x', softplus=False, need_r1_and_x1=True)
    else:
        def x(d, optimizer=None, var=None):
            if x_use == 'x':
                out = _get_r1_and_x1(optimizer, var=var)
                return out[-1]
            else:
                return d

    assert y_use in ['rnn', 'none']
    if y_use == 'rnn':
        y = generate_rnn_optimizer(name, 'y', softplus=True)
    else:
        def y(*args, optimizer=None, var=None):
            return tf.ones_like(args[0])

    d = generate_rnn_optimizer(name, 'd', softplus=True, need_r1_and_x1=True)

    sf = generate_rnn_state_transform()
    return {'x': x, 'd': d, 'y': y, 'sf': sf}


def RNN_optimizers(**kwargs):
    return {'conv2d/kernel': generate_x_y_sf('conv2d/kernel', **kwargs),
            'conv2d/bias': generate_x_y_sf('conv2d/bias', **kwargs),
            'batch/gamma': generate_x_y_sf('batch/gamma', **kwargs),
            'batch/beta': generate_x_y_sf('batch/beta', **kwargs),
            'dense/kernel': generate_x_y_sf('dense/kernel', **kwargs),
            'dense/bias': generate_x_y_sf('dense/bias', **kwargs)}
