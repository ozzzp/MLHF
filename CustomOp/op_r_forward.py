import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import mean, rsqrt_grad
from tensorflow.python.ops.gen_nn_ops import max_pool_grad_grad
from tensorflow.python.ops.math_ops import sigmoid_grad
from tensorflow.python.ops.nn_grad import _ReluGrad


def conv2d_forward(op, r_input):
    with tf.name_scope('conv2d_forward'):
        input = list(op.inputs)
        data_format = op.get_attr('data_format')
        padding = op.get_attr('padding')
        strides = op.get_attr('strides')

        p1 = tf.nn.conv2d(input[0], r_input[1], strides=strides, padding=padding, data_format=data_format)
        p2 = tf.nn.conv2d(r_input[0], input[1], strides=strides, padding=padding, data_format=data_format)

        return [p1 + p2]


def identity_forward(op, r_inpput):
    return [tf.identity(r_inpput[0])]


def relu_forward(op, r_input):
    with tf.name_scope('relu_forward'):
        return [_ReluGrad(op, r_input[0])]


def add_forward(op, r_input):
    with tf.name_scope('add_forward'):
        return [r_input[0] + r_input[1]]


def pad_forward(op, r_input):
    with tf.name_scope('pad_forward'):
        return [tf.pad(r_input[0], op.inputs[1])]


def avgpool_forward(op, r_input):
    with tf.name_scope('avgpool_forward'):
        data_format = op.get_attr('data_format')
        padding = op.get_attr('padding')
        strides = op.get_attr('strides')
        ksize = op.get_attr('ksize')
        return [tf.nn.avg_pool(r_input[0], data_format=data_format, padding=padding, strides=strides, ksize=ksize)]


def maxpool_forward(op, r_input):
    with tf.name_scope('maxpool_forward'):
        data_format = op.get_attr('data_format')
        padding = op.get_attr('padding')
        strides = op.get_attr('strides')
        ksize = op.get_attr('ksize')
        return [max_pool_grad_grad(op.inputs[0],
                                   op.outputs[0],
                                   r_input[0], data_format=data_format, padding=padding, strides=strides, ksize=ksize)]


def reshape_forward(op, r_input):
    return [tf.reshape(r_input[0], op.inputs[1])]


def matmul_forward(op, r_input):
    with tf.name_scope('matmul_forward'):
        input = list(op.inputs)
        transpose_a = op.get_attr('transpose_a')
        transpose_b = op.get_attr('transpose_b')

        p1 = tf.matmul(input[0], r_input[1], transpose_a=transpose_a, transpose_b=transpose_b)
        p2 = tf.matmul(r_input[0], input[1], transpose_a=transpose_a, transpose_b=transpose_b)
        return [p1 + p2]


def biasadd_forward(op, r_input):
    with tf.name_scope('bias_forward'):
        data_format = op.get_attr('data_format')
        return [tf.nn.bias_add(r_input[0], r_input[1], data_format=data_format)]


def const_forward(op, r_input):
    return [tf.zeros_like(op.outputs[0])]


def mean_forward(op, r_input):
    with tf.name_scope('mean_forward'):
        keep_dims = op.get_attr('keep_dims')
        return [mean(r_input[0], op.inputs[1], keep_dims=keep_dims)]


def squeeze_forward(op, r_input):
    with tf.name_scope('squeeze_forward'):
        squeeze_dims = op.get_attr('squeeze_dims')
        return [tf.squeeze(r_input[0], axis=squeeze_dims)]


def stopgradient_forward(op, r_input):
    return [tf.zeros_like(op.inputs[0])]


def squaredifference_forward(op, r_input):
    with tf.name_scope('squaredifference_forward'):
        x_minus_y_times_2 = 2 * (op.inputs[0] - op.inputs[1])
        return [x_minus_y_times_2 * (r_input[0] - r_input[1])]


def rsqrt_forward(op, r_input):
    # y = x^(-0.5)
    # dy = -0.5*x^(-1.5)dx = -0.5*y^3dx
    with tf.name_scope('rsqrt_forward'):
        return [rsqrt_grad(op.outputs[0], r_input[0])]


def mul_forward(op, r_input):
    with tf.name_scope('mul_forward'):
        p1 = op.inputs[0] * r_input[1]
        p2 = r_input[0] * op.inputs[1]
        return [p1 + p2]


def sub_forward(op, r_input):
    with tf.name_scope('sub_forward'):
        return [r_input[0] - r_input[1]]


def enter_forward(op, r_input):
    return [r_input[0]]


def sigmoid_forward(op, r_input):
    with tf.name_scope('sigmoid_forward'):
        return sigmoid_grad(op.outputs[0], r_input[0])


op_r_forward_funcs = {'Conv2D': conv2d_forward,
                      'Identity': identity_forward,
                      'Relu': relu_forward,
                      'Add': add_forward,
                      'Pad': pad_forward,
                      'AvgPool': avgpool_forward,
                      'MaxPool': maxpool_forward,
                      'Reshape': reshape_forward,
                      'MatMul': matmul_forward,
                      'BiasAdd': biasadd_forward,
                      'Const': const_forward,
                      'Mean': mean_forward,
                      'Squeeze': squeeze_forward,
                      'StopGradient': stopgradient_forward,
                      'SquaredDifference': squaredifference_forward,
                      'Rsqrt': rsqrt_forward,
                      'Mul': mul_forward,
                      'Sub': sub_forward,
                      'Enter': enter_forward,
                      'Sigmoid': sigmoid_forward}

if __name__ == '__main__':
    learning_rate = 1
    import CustomOp as co
    from models.official.resnet.resnet_model import cifar10_resnet_v2_generator
    from CustomOp.hession_loss import loss_types

    inputs = tf.random_normal([128, 32, 32, 3], stddev=5)

    network = cifar10_resnet_v2_generator(32, 10, 'channels_last')

    # Batch norm requires update ops to be added as a dependency to the train_op
    with tf.variable_scope('train_net', reuse=tf.AUTO_REUSE):
        logits = network(inputs, True)
    optimizer = co.MetaHessionFreeOptimizer(learning_rate=learning_rate)

    # Batch norm requires update ops to be added as a dependency to the train_op

    vars = tf.trainable_variables(scope='train_net')

    rs1 = [tf.random_normal(v.get_shape().as_list()) for v in vars]
    k1 = tf.random_uniform([], maxval=100)
    rs2 = [tf.random_normal(v.get_shape().as_list()) for v in vars]
    k2 = tf.random_uniform([], maxval=100)
    rsn = [k1 * r1 + k2 * r2 for r1, r2 in zip(rs1, rs2)]
    loss_fn, Hv_fun = loss_types['cross_entropy']
    Hv = optimizer._generate_Hv_fun(var_list=vars, out=logits, input_list=[inputs], Hl_func=Hv_fun)
    # Hv._python_grad_func = None
    rds1 = list(Hv(*rs1))
    rds2 = tf.gradients(ys=rds1, xs=rs1, grad_ys=rs1)

    A = tf.global_norm(rds1)
    B = tf.global_norm(rds2)
    delta = tf.global_norm([rd1 - rd2 for rd1, rd2 in zip(rds1, rds2)])
    loss = delta / A

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            out = sess.run([loss, delta, A, B])
            print(out)
            pass
