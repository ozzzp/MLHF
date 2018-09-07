import itertools
# import numpy as np
import os

import tensorflow as tf
import tensorflow.contrib as tfcb
from tensorflow.python.framework import function
from tensorflow.python.training.optimizer import _var_key
from tensorflow.python.training.slot_creator import create_slot_with_initializer

from .gradients_impl import gradients
from .hession_loss import loss_types
from .op_r_forward import op_r_forward_funcs
from .rnn import RNN_optimizers, slot_name, _EPSILON


class MetaHessionFreeOptimizer(tf.train.GradientDescentOptimizer):
    def __init__(self, learning_rate, optimizers=RNN_optimizers, is_training=False, use_locking=False,
                 name="MetaHessionFree", iter=5, damping=2e-5, damping_type='regular', decay=2 / 3, print_log=False,
                 **kwargs):
        self._optimizers = optimizers(**kwargs)
        self._is_training = is_training
        self._n = iter
        self._print_log = print_log
        self._damping = damping
        self._decay = decay
        assert damping_type in ['regular', 'LM_heuristics']
        self._damping_type = damping_type
        super(MetaHessionFreeOptimizer, self).__init__(learning_rate=learning_rate,
                                                       use_locking=use_locking,
                                                       name=name)

    @staticmethod
    def _r_forward(r_v_list, out, input_list):
        with tf.name_scope('difference_forward'):
            r_dict = {v.value(): r for r, v in r_v_list}
            r_dict.update({input: tf.zeros_like(input) for input in input_list})
            used_ops = tfcb.graph_editor.get_backward_walk_ops(seed_ops=out.op, stop_at_ts=list(r_dict.keys()))
            used_ops = reversed(used_ops)

            while True:
                last_ops = []
                for op in used_ops:
                    has_floating = set(i.dtype.is_floating for i in op.outputs)
                    if True not in has_floating:
                        outs = [None for i in op.outputs]
                    else:
                        try:
                            r_input = [r_dict[i] for i in op.inputs]
                        except:
                            last_ops.append(op)
                            continue
                        assert op.type in op_r_forward_funcs, op.type
                        forward_func = op_r_forward_funcs[op.type]
                        outs = forward_func(op, r_input)
                    r_dict.update({v: r for r, v in zip(outs, op.outputs)})
                if last_ops:
                    used_ops = last_ops
                else:
                    break

            assert out in r_dict
            return r_dict[out]

    def _generate_Hv_fun(self, var_list, out, input_list, Hl_func, ds=None, damping=0):
        def shape_func(op):
            return [var.get_shape() for var in var_list]

        if ds is not None:
            dampings = [self._generate_d(d, var=v) + damping for d, v in zip(ds, var_list)]
        else:
            dampings = None

        @function.Defun(*[v.dtype for v in var_list], shape_func=shape_func)
        def Hv(*vs):
            assert len(var_list) == len(vs)
            for var, v in zip(var_list, vs):
                v.set_shape(var.get_shape())
            with tf.name_scope('Hession_product', values=vs):
                # difference forward
                r_out = self._r_forward(r_v_list=list(zip(vs, var_list)), out=out, input_list=input_list)
                print('difference forward done')

                rd_out = Hl_func(r_out, out)
                # TODO define RNN #To stable Hession
                # Oops, still no idea.

                # difference backword, same as common back propagation but with special init grad.

                rds = self._rd_backward(out=out, rd_out=rd_out, v_list=var_list)
                print('difference backward done')
                '''
                test_case = set(tf.gradients(rds, var_list))
                assert test_case == {None}
                test_case = set(tf.gradients(rds, vs))
                assert None not in test_case
                '''
                return tuple(rds)

        def grad_Hv(op, *vs):
            Hv_extra_inputs_backup = Hv._extra_inputs
            Hv._extra_inputs = list(op.inputs)[len(vs):]
            outs = list(Hv(*vs))
            nones = [None] * (len(op.inputs) - len(outs))
            Hv._extra_inputs = Hv_extra_inputs_backup
            return tuple(outs + nones)

        Hv._python_grad_func = grad_Hv

        def _Hv(*vs):
            rds = Hv(*vs)
            if dampings is not None:
                rds = [rd + damping * v for rd, damping, v in zip(rds, dampings, vs)]
            return tuple(rds)

        return _Hv

    def _get_or_make_slot_with_initializer(self, var, initializer, shape, dtype,
                                           slot_name, op_name):
        """Find or create a slot for a variable, using an Initializer.

        Args:
          var: A `Variable` object.
          initializer: An `Initializer`.  The initial value of the slot.
          shape: Shape of the initial value of the slot.
          dtype: Type of the value of the slot.
          slot_name: Name for the slot.
          op_name: Name to use when scoping the Variable that
            needs to be created for the slot.

        Returns:
          A `Variable` object.
        """
        named_slots = self._slot_dict(slot_name)
        if _var_key(var) not in named_slots:
            with tf.variable_scope('slots', reuse=tf.AUTO_REUSE):
                named_slots[_var_key(var)] = create_slot_with_initializer(
                    var, initializer, shape, dtype, op_name)
        return named_slots[_var_key(var)]

    @staticmethod
    def _inner_product(A_list, B_list):
        sum_list = [tf.reduce_sum(A * B) for A, B in zip(A_list, B_list)]
        return tf.add_n(sum_list)

    def _generate_x(self, d, var=None):
        with tf.name_scope('rnn_x'):
            name = os.path.join(*[i.split('_')[0] for i in var.op.name.rsplit('/', 3)[-2:]])
            assert name in self._optimizers, 'sorry, rnn optimizer of {} is not defined'.format(name)
            x_fn = self._optimizers[name]['x']
            out = x_fn(d, var=var, optimizer=self)
            return out

    def _generate_d(self, d, var=None):
        with tf.name_scope('rnn_d'):
            name = os.path.join(*[i.split('_')[0] for i in var.op.name.rsplit('/', 3)[-2:]])
            assert name in self._optimizers, 'sorry, rnn optimizer of {} is not defined'.format(name)
            d_fn = self._optimizers[name]['d']
            out = d_fn(d, var=var, optimizer=self)
            return out

    def _generate_state_transform(self, r_1, x_1, var=None):
        with tf.name_scope('rnn_sf'):
            name = os.path.join(*[i.split('_')[0] for i in var.op.name.rsplit('/', 3)[-2:]])
            assert name in self._optimizers, 'sorry, rnn state transform of {} is not defined'.format(name)
            sf_fn = self._optimizers[name]['sf']
            sf_fn(r_1=r_1, x_1=x_1, var=var, optimizer=self)

    def _generate_y(self, d, r_0, x_0, var=None):
        with tf.name_scope('rnn_y'):
            name = os.path.join(*[i.split('_')[0] for i in var.op.name.rsplit('/', 3)[-2:]])
            assert name in self._optimizers, 'sorry, rnn optimizer of {} is not defined'.format(name)
            y_fn = self._optimizers[name]['y']
            out = y_fn(d, r_0, x_0, var=var, optimizer=self)
            return out

    def _rd_backward(self, out, rd_out, v_list):
        rd_list = gradients(out, v_list, grad_ys=rd_out)
        return rd_list

    def compute_gradients(self, *args, **kwargs):
        raise NotImplementedError("Sorry, call compute_gradients directly is not allowed")

    def _compute_gradients(self, *args, **kwargs):
        return super(MetaHessionFreeOptimizer, self).compute_gradients(*args,
                                                                       gate_gradients=MetaHessionFreeOptimizer.GATE_NONE,
                                                                       **kwargs)

    def apply_gradients(self, *args, **kwargs):
        raise NotImplementedError("Sorry, call compute_gradients directly is not allowed")

    def _apply_gradients(self, *args, **kwargs):
        return super(MetaHessionFreeOptimizer, self).apply_gradients(*args, **kwargs)

    def set_slot_shadow(self, var, val, slot_name, replace=False):
        named_slots = self._slot_dict(slot_name + '_shadow')
        key = var if isinstance(var, str) else _var_key(var)

        if replace:
            assert key in named_slots
        else:
            assert key not in named_slots
        named_slots[key] = val

    def _apply_state(self, var):
        ops = []
        for rnn_type in ['x', 'y', 'd']:
            for l in itertools.count():
                for i in itertools.count():
                    slot_var = self.get_slot(var, slot_name(l, i, rnn_type))
                    if slot_var is None:
                        break
                    slot_val = self.get_slot(var, slot_name(l, i, rnn_type) + '_shadow')
                    if self._is_training:
                        ops.append((slot_var, slot_val))
                    else:
                        ops.append(tf.assign(slot_var, slot_val))
                if i == 0:
                    break
        for val_name in ['r_1', 'x_1']:
            slot_var = self.get_slot(var, val_name)
            slot_val = self.get_slot(var, val_name + '_shadow')
            assert isinstance(slot_var, tf.Variable)
            assert isinstance(slot_val, tf.Tensor)
            if self._is_training:
                ops.append((slot_var, slot_val))
            else:
                ops.append(tf.assign(slot_var, slot_val))
        return ops

    def _apply_dense(self, grad, var):
        ops = self._apply_state(var)
        with tf.control_dependencies(ops):
            return super(MetaHessionFreeOptimizer, self)._apply_dense(grad, var)

    def _apply_sparse(self, grad, var):
        raise NotImplementedError

    def _resource_apply_dense(self, grad, handle):
        raise NotImplementedError

    def _resource_apply_sparse(self, grad, handle, indices):
        raise NotImplementedError

    def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
        raise NotImplementedError

    def minimize(self, loss_type, out, label, input_list, global_step=None, var_list=None, network_fn=None):
        assert loss_type in loss_types
        loss_fn, Hl_fun = loss_types[loss_type]

        # 1st forward
        loss = loss_fn(out, label)
        print('1st forward done')

        # 2nd backward
        d_and_v = self._compute_gradients(loss, var_list=var_list)
        print('2nd backward done')

        if self._damping_type == 'LM_heuristics':
            assert callable(network_fn)
            self._last_loss = tf.get_variable('last_loss', initializer=tf.zeros_initializer, shape=[], dtype=tf.float32)
            self._q_difference = tf.get_variable('q_difference', initializer=tf.zeros_initializer, shape=[],
                                                 dtype=tf.float32)
            self._last_inputs = [
                tf.get_variable('last_input_{}'.format(i), initializer=tf.zeros_initializer, shape=input.shape,
                                dtype=input.dtype, trainable=False)
                for i, input in enumerate(input_list)]
            self._last_label = tf.get_variable('last_label', initializer=tf.zeros_initializer, shape=label.shape,
                                               dtype=label.dtype, trainable=False)
            self._damping = tf.get_variable('damping', initializer=self._damping, dtype=tf.float32,
                                            trainable=False)

            loss_on_last_batch = loss_fn(network_fn(*self._last_inputs), self._last_label)
            rho = (
                              loss_on_last_batch - self._last_loss) / self._q_difference  # tf.Print(self._q_difference, [self._q_difference, loss_on_last_batch, self._last_loss])
            rho = tf.where(tf.equal(self._q_difference, 0), 0.5, rho)
            # rho = tf.Print(rho, [rho], message='rho:')
            decay = tf.train.piecewise_constant(rho, [0.25, 0.75], [1 / self._decay, 1., self._decay])
            # decay = tf.Print(decay, [decay], message='decay:')
            damping = self._damping * decay
            damping = tf.clip_by_value(damping, 1e-3, 1)
            # damping = tf.Print(damping, [damping], message='damping:')
        else:
            damping = self._damping

        ds = [tf.stop_gradient(d) for d, _ in d_and_v if d is not None]
        if not ds:
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops"
                " that do not support gradients, between variables %s and loss %s." %
                ([str(v) for _, v in d_and_v], loss))

        var_list = [v for d, v in d_and_v if d is not None]
        Hv_fun = self._generate_Hv_fun(ds=ds, var_list=var_list, out=out, input_list=input_list, Hl_func=Hl_fun,
                                       damping=damping)

        # generate x_0 from (d, r_1, x_1)
        x_is = [self._generate_x(d, var=v) for d, v in zip(ds, var_list)]
        print('rnn_x generated')

        H_xis = list(Hv_fun(*x_is))

        # r_0 = b - H_x0 = d - H_x0
        ds_length = tf.global_norm(ds)
        ds_length_sq = ds_length ** 2
        # gamma_0 = self._inner_product(ds, H_x0s) / ds_length_sq
        r_is = [d - H_xi for d, H_xi in zip(ds, H_xis)]

        # y_0 = r_0 * p
        #  p = f(r_0, x_0, d)
        # so, y_0 =r_0 * f(r_0, x_0, d)
        Ps = [self._generate_y(d, r_i, x_i, var=v) for d, r_i, x_i, v in
              zip(ds, r_is, x_is, var_list)]
        print('rnn_y generated')

        y_is = [P * r_i for P, r_i in zip(Ps, r_is)]

        p_is = y_is

        beta_part = self._inner_product(r_is, y_is)

        def _cal(p_is, r_is, x_is, beta_part):
            # y_0 as p_0
            # cal H_p0 = H_y0
            H_pis = list(Hv_fun(*p_is))

            # \alpha = <r_0, y_0>/<p0 , H_p0> = <r_0, y_0>/<y_0 , H_y0>
            p2 = self._inner_product(p_is, H_pis)
            alpha = beta_part / tf.maximum(p2, _EPSILON)

            # x_1 = x_0 + \alpha p_0 = x_0 + \alpha y_0
            x_is = [x_i + alpha * p_i for x_i, p_i in zip(x_is, p_is)]

            # r_1 = r_0 - \alpha H_p0 = r_0 - \alpha H_y0
            # gamma_1 = self._inner_product(ds, H_y0s) / ds_length_sq
            r_is = [r_i - alpha * H_pi for r_i, H_pi in zip(r_is, H_pis)]

            y_is = [P * r_i for P, r_i in zip(Ps, r_is)]

            new_beta_part = self._inner_product(r_is, y_is)
            beta = new_beta_part / tf.maximum(beta_part, _EPSILON)
            beta_part = new_beta_part

            p_is = [y_i + beta * p_i for y_i, p_i in zip(y_is, p_is)]

            return p_is, r_is, x_is, beta_part

        def _cond(p_is, r_is, x_is, beta_part):
            return tf.global_norm(r_is) >= _EPSILON

        loop_vars = (p_is, r_is, x_is, beta_part)

        p_is, r_is, x_is, beta_part = \
            tf.while_loop(_cond, _cal, loop_vars, swap_memory=True,
                          back_prop=self._is_training,
                          parallel_iterations=1,
                          maximum_iterations=self._n)

        # apply state transform.
        for r_i, x_i, var in zip(r_is, x_is, var_list):
            self._generate_state_transform(r_i, x_i, var=var)
        print('rnn_sf generated')

        inner_p_ds_x_is = self._inner_product(ds, x_is)
        H_xis = [d - r_i for d, r_i in zip(ds, r_is)]
        x_is_H_xis = self._inner_product(x_is, H_xis)
        if self._damping_type == 'LM_heuristics':
            q_difference = - self._learning_rate * inner_p_ds_x_is + self._learning_rate ** 2 / 2 * x_is_H_xis

        if self._is_training:

            hession_loss = - inner_p_ds_x_is / tf.sqrt(x_is_H_xis)
            # minize r_1
            # assert there should be no grad which would backprop from x_1s to nn variable.
            r_loss = tf.global_norm(r_is)
            var_length = tf.stop_gradient(tf.global_norm(var_list))

            if self._print_log:
                H_ds = list(Hv_fun(*ds))
                standard_loss = ds_length_sq / tf.sqrt(self._inner_product(ds, H_ds))
                hession_loss = tf.Print(hession_loss,
                                        [tf.global_norm(x_is) / ds_length, r_loss, -hession_loss,
                                         standard_loss, inner_p_ds_x_is,
                                         ds_length_sq,
                                         var_length, loss],
                                        message='x1l/gl, rl, hs, ss, hip, sip, vl, loss:')

            x_is, _ = tf.clip_by_global_norm(x_is, var_length * (0.25 / self._learning_rate))
            next_state = []
            for x_i, v in zip(x_is, var_list):
                noise = tf.random_uniform(x_i.get_shape(), 1 - 2e-2, 1 + 2e-2)
                next_state.append((v, tf.stop_gradient(v) - noise * self._learning_rate * x_i))
                next_state.extend(self._apply_state(v))
            assert len(next_state) == len(tf.global_variables(scope='slots')) + len(tf.trainable_variables(scope='nn'))
            if self._damping_type == 'LM_heuristics':
                for val, var in zip(input_list, self._last_inputs):
                    next_state.append((var, val))
                next_state.append((self._damping, damping))
                next_state.append((self._last_loss, loss))
                next_state.append((self._last_label, label))
                next_state.append((self._q_difference, q_difference))
            return next_state, loss, hession_loss, r_loss
        else:
            if self._damping_type == 'LM_heuristics':
                depends = [tf.assign(var, val) for val, var in zip(input_list, self._last_inputs)]
                depends.append(tf.assign(self._damping, damping))
                depends.append(tf.assign(self._last_loss, loss))
                depends.append(tf.assign(self._last_label, label))
                depends.append(tf.assign(self._q_difference, q_difference))
            else:
                depends = []
            with tf.control_dependencies(depends):
                return self._apply_gradients(list(zip(x_is, var_list)), global_step=global_step)
