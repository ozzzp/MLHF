import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.basic_session_run_hooks import SessionRunArgs

from .MetaOptimizer import MetaHessionFreeOptimizer
from .hession_loss import loss_types


def debug_grad():
    from unittest.mock import _patch as patch
    block_ops = {'Identity', 'Enter', 'Exit', 'Switch', 'Tile',
                 'NextIteration', 'Merge', 'Reshape', 'Select',
                 'Pack', 'Slice', 'Squeeze', 'Pad', 'TensorArrayGatherV3',
                 'TensorArrayWriteV3', 'Print', 'StridedSlice'}

    ops_type = set()

    def check_numerics_ops(outs, op):
        def verify_tensor_all_finite(ts):
            if op.type not in block_ops and ts is not None and ts.dtype.is_floating:
                out_ts = tf.verify_tensor_all_finite(ts, msg="Grad oops: " + op.name)
                if op.type not in ops_type:
                    print('debug op: {}'.format(op.type))
                    ops_type.add(op.type)
            else:
                out_ts = ts
            return out_ts

        if outs is None or isinstance(outs, tf.Tensor):
            outs = verify_tensor_all_finite(outs)
        else:
            outs = [verify_tensor_all_finite(i) for i in outs]
        return outs

    def custom_MaybeCompile(scope, op, func, grad_fn):
        outs = grad_fn()
        outs = check_numerics_ops(outs, op)
        return outs

    return patch(getter=lambda: __import__("tensorflow.python.ops.gradients_impl", fromlist=['_MaybeCompile']),
                 attribute='_MaybeCompile', new=custom_MaybeCompile,
                 spec=None, create=False, spec_set=None, autospec=None, new_callable=None, kwargs={})


class varible_replace_record:
    def __init__(self, var, replaced=None):
        self._var = var
        assert replaced is not None
        self._replaced = replaced

    def __enter__(self):
        self._old_consumers = [self._get_variable_consumers(v, loop_anchor=rep) for v, rep in
                               zip(self._var, self._replaced)]

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert exc_type is None
        self._modify_consumer = [list(set(self._get_variable_consumers(v, loop_anchor=rep)) - set(old_consumer))
                                 for v, rep, old_consumer in zip(self._var, self._replaced, self._old_consumers)]
        self.replace_varible(self._replaced)

    def replace_varible(self, new_list):
        assert len(new_list) == len(self._var)
        for new_var, modify_ops in zip(new_list, self._modify_consumer):
            for op, v_replaced in modify_ops:
                ge.reroute_ts(new_var, v_replaced, can_modify=[op])

    @staticmethod
    def _get_variable_consumers(var, loop_anchor):
        assert isinstance(var, tf.Variable)
        assert isinstance(loop_anchor, tf.Tensor)

        def A_develop_to_B(A, B):
            if isinstance(A, tf.Operation):
                A = A._control_flow_context
            if isinstance(B, tf.Operation):
                B = B._control_flow_context
            if B == A:
                return True
            if B is None:
                return False
            else:
                return A_develop_to_B(A, B.outer_context)

        consumers = []
        ops = [(i, var._variable) for i in var._variable.consumers()]
        while True:
            last_ops = []
            for op, ts in ops:
                if op.type == 'Identity' and op.name.rsplit('/', 2)[-1] == 'read':
                    last_ops.extend([(i, op.outputs[0]) for i in op.outputs[0].consumers()])
                elif op.type == 'Enter':
                    if A_develop_to_B(op, loop_anchor.op):
                        last_ops.extend([(i, op.outputs[0]) for i in op.outputs[0].consumers()])
                    elif A_develop_to_B(loop_anchor.op, op):
                        consumers.append((op, ts))
                else:
                    if loop_anchor.op._control_flow_context == op._control_flow_context:
                        consumers.append((op, ts))
            if len(last_ops) == 0:
                break
            else:
                ops = last_ops
        return consumers


def _build_network(network, inputs, labels, params, varible_list=None, need_optimizer=True):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Enter': 'Enter_custom'}):
        with tf.variable_scope('nn', reuse=tf.AUTO_REUSE):
            logits = network(inputs, True)
    if need_optimizer:
        optimizer = MetaHessionFreeOptimizer(learning_rate=params['lr'],
                                             is_training=True,
                                             iter=params['CG_iter'],
                                             print_log=params['print_log_in_meta_optimizer'],
                                             x_use=params['x_use'],
                                             y_use=params['y_use'])
        var_state, loss, hession_loss, r_loss = optimizer.minimize(loss_type='cross_entropy',
                                                                   out=logits,
                                                                   label=labels,
                                                                   input_list=[inputs],
                                                                   global_step=None,
                                                                   var_list=varible_list)
        return var_state, loss, hession_loss, r_loss
    else:
        loss_fn, Hl_fun = loss_types['cross_entropy']
        loss = loss_fn(logits, labels)
        return loss


def build_meta_train_rnn(network, inputs, labels, params):
    var_state, loss, hession_loss, r_loss = _build_network(network, inputs[0], labels[0], params, need_optimizer=True)
    var_list = [v for v, _ in var_state]
    print('generated sample network.')

    def _cal(i, loss_ts_array, test_loss_ts_array, hession_loss_ts_array, r_loss_ts_array, *var):
        with varible_replace_record(var=var_list, replaced=var):
            next_state, next_loss, next_hession_loss, next_r_loss = _build_network(network,
                                                                                   inputs[i], labels[i], params,
                                                                                   varible_list=[v for v in var_list if
                                                                                                 v in tf.trainable_variables()],
                                                                                   need_optimizer=True)
        print('generated network in loop.')
        with varible_replace_record(var=var_list,
                                    replaced=[new_v if v in tf.trainable_variables() else old_v for old_v, (v, new_v) in
                                              zip(var, next_state)]):
            next_loss_2 = _build_network(network, inputs[i], labels[i], params,
                                         varible_list=[v for v in var_list if v in tf.trainable_variables()],
                                         need_optimizer=False)
        loss_ts_array = loss_ts_array.write(i, next_loss)
        test_loss_ts_array = test_loss_ts_array.write(i, next_loss_2)
        hession_loss_ts_array = hession_loss_ts_array.write(i, next_hession_loss)
        r_loss_ts_array = r_loss_ts_array.write(i, next_r_loss)
        return tuple(
            [i + 1, loss_ts_array, test_loss_ts_array, hession_loss_ts_array, r_loss_ts_array] + [ns for _, ns in
                                                                                                  next_state])

    n = tf.shape(labels)[0]

    def _cond(i, *arg):
        return i < n

    loop_vars = tuple([tf.constant(0, tf.int32),
                       tf.TensorArray(tf.float32, size=n),
                       tf.TensorArray(tf.float32, size=n),
                       tf.TensorArray(tf.float32, size=n),
                       tf.TensorArray(tf.float32, size=n)] + \
                      [v.value() for v in var_list])

    out = tf.while_loop(_cond, _cal, loop_vars, swap_memory=True,
                        back_prop=True,
                        parallel_iterations=1,
                        maximum_iterations=n)

    def update_fn():
        with tf.control_dependencies([v.assign(nv) for v, nv in zip(var_list, out[5:])]):
            return tf.no_op()


    loss = out[1].stack()
    test_loss = out[2].stack()
    hession_loss = out[3].stack()
    r_loss = out[4].stack()

    base_loss = loss[-1]
    tf.identity(base_loss, name='base_loss')
    tf.summary.scalar('base_loss', base_loss)

    meta_loss_part_1 = tf.reduce_mean(loss[1:] - loss[:-1])
    tf.identity(meta_loss_part_1, name='meta_loss_part_1')
    tf.summary.scalar('meta_loss_part_1', meta_loss_part_1)

    meta_loss_part_2 = tf.reduce_mean(test_loss - loss)
    tf.identity(meta_loss_part_2, name='meta_loss_part_2')
    tf.summary.scalar('meta_loss_part_2', meta_loss_part_2)

    meta_loss = (test_loss[:-1] + loss[1:]) / 2 - tf.stop_gradient(loss[:-1])
    meta_loss = tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(meta_loss)) * meta_loss)
    tf.identity(meta_loss, name='meta_loss')
    tf.summary.scalar('meta_loss', meta_loss)

    hession_loss = tf.reduce_mean(hession_loss)
    tf.identity(hession_loss, name='hession_loss')
    tf.summary.scalar('hession_loss', hession_loss)

    r_loss = tf.reduce_mean(r_loss)
    tf.identity(r_loss, name='r_loss')
    tf.summary.scalar('r_loss', r_loss)

    return base_loss, meta_loss, hession_loss, r_loss, update_fn


class checkpoint_loader(object):
    """
    one who understands .ckpt files, very much
    """

    def __init__(self, *args):
        self.src_key = list()
        self.vals = list()
        self.load(*args)

    def __call__(self, key):
        # for idx in range(len(key)):
        idx = 0
        val = self.find(key, idx)
        if val is not None:
            return val
        else:
            return None

    def find(self, key, idx):
        up_to = len(self.src_key)
        for i in range(up_to):
            key_b = self.src_key[i]
            if key_b[idx:] == key[idx:]:
                return self.yields(i)
        return None

    def yields(self, idx):
        del self.src_key[idx]
        temp = self.vals[idx]
        del self.vals[idx]
        return temp

    def load(self, ckpt):
        from tensorflow.python import pywrap_tensorflow
        reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for name, shape in sorted(var_to_shape_map.items()):
            # print("find: ", name)
            packet = [name, shape]
            self.src_key += [packet]
            self.vals += [reader.get_tensor(name)]


def load_old_graph(sess, ckpt, varibles=None):
    ckpt_loader = checkpoint_loader(ckpt)

    if varibles is None:
        varibles = tf.global_variables()
    for i, var in enumerate(varibles):
        name = var.name.split(':')[0]
        args = [name, var.get_shape()]
        val = ckpt_loader(args)
        if val is None:
            raise ValueError('{}/{}: Cannot find and load {}'.format(i, len(tf.global_variables()), args))
        else:
            # print("{}/{}: loading {};".format(i, len(tf.global_variables()), var.name))
            var.load(val, sess)


def load_meta_parameters(sess, model_dir, scope='metaoptimizer'):
    ckpt = tf.train.get_checkpoint_state(model_dir)
    varibles = tf.global_variables(scope=scope)
    if len(varibles) > 0:
        if ckpt is not None:
            load_old_graph(sess, ckpt.model_checkpoint_path, varibles=varibles)
            step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[-1])
            print('Restored from meta ckpt {}...'.format(step))
            return step
        else:
            return None
    else:
        return True


class MetaParametersLoadingHook(session_run_hook.SessionRunHook):
    def __init__(self, path=''):
        self._path = path
        self._first_run = True

    def before_run(self, run_context):  # pylint: disable=unused-argument
        if self._first_run:
            out = load_meta_parameters(run_context.session, self._path)
            assert out is not None, "Sorry, path {} not founded!".format(self._path)
            self._first_run = False
        return SessionRunArgs({})

    def after_run(self, run_context, run_values):
        pass


def merge_grad_and_vars(*args):
    var_to_grad = {}
    for arg in args:
        assert isinstance(arg, list)
        for grad, var in arg:
            if grad is None:
                continue
            if var not in var_to_grad:
                var_to_grad[var] = grad
            else:
                var_to_grad[var] += grad
    return [(grad, var) for var, grad in var_to_grad.items()]
