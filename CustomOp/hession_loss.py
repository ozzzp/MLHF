import tensorflow as tf


def H_l_mul_v(v, x):
    softmax = tf.nn.softmax(x)
    out = v - tf.reduce_sum(softmax * v, 1, keepdims=True)
    out = softmax * out / tf.to_float(tf.shape(x)[0])
    out = out - tf.reduce_sum(out, 1, keepdims=True) * softmax
    return out


# {'loss_type': (loss func, Hv func), ...}
loss_types = {'cross_entropy': (lambda x, y: tf.losses.softmax_cross_entropy(logits=x, onehot_labels=y),
                                H_l_mul_v),
              'l2_loss': (lambda x, y: tf.nn.l2_loss(x - y) / tf.to_float(tf.shape(x)[0]),
                          lambda v, x: v / tf.to_float(tf.shape(x)[0]))}

if '__main__' == __name__:
    for loss_func, hv_func in loss_types.values():
        def get_hession_matrix_mutiply_v(v, x):
            loss = loss_func(x, tf.stop_gradient(x))
            old_shape = v.get_shape()
            num_elements = old_shape.num_elements()
            H = tf.hessians(loss, x)
            H = tf.reshape(H, [num_elements, num_elements])
            v = tf.reshape(v, [num_elements, 1])
            out = tf.matmul(H, v)
            return tf.reshape(out, old_shape.as_list()), H


        def verify_result(v, x):
            standard, H = get_hession_matrix_mutiply_v(v, x)
            tested = hv_func(v, x)
            p1 = tf.sqrt(2 * tf.nn.l2_loss(standard))
            p2 = tf.sqrt(2 * tf.nn.l2_loss(tested))
            delta = tf.sqrt(2 * tf.nn.l2_loss(tested - standard))

            delta = tf.Print(delta, [delta / p1, delta, p1, p2])
            with tf.control_dependencies([delta.op]):
                H = tf.identity(H)
            return H


        X = tf.random_normal([64, 100])
        V = tf.random_normal([64, 100])
        H = verify_result(V, X)
        with tf.Session() as sess:
            out = sess.run(H)
            pass
