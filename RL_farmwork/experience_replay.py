import os
from multiprocessing.managers import BaseManager, BaseProxy

import numpy as np
import tensorflow as tf
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.basic_session_run_hooks import SessionRunArgs

from rank_based import Experience


class sync_expericence(Experience):
    def __init__(self, conf, path):
        super(sync_expericence, self).__init__(conf)
        assert isinstance(path, str)
        self._path = os.path.join(path, 'experience_replay.npy')
        self.load()

    def store(self, experience):
        out = super(sync_expericence, self).store(experience)
        self.save()
        return out

    def load(self):
        if os.path.exists(self._path):
            super(sync_expericence, self).load(self._path)
            print('load experience from {}'.format(self._path))
        else:
            pass

    def save(self):
        super(sync_expericence, self).save(self._path)

    def sample(self, global_step, batch_size=None):
        return super(sync_expericence, self).sample(global_step, batch_size=batch_size)

    def update_priority(self, indices, delta):
        super(sync_expericence, self).update_priority(indices, delta)
        super(sync_expericence, self).rebalance()


class ExperienceProxy(BaseProxy):
    _exposed_ = ('store', 'sample', 'update_priority', 'load', 'save')

    def store(self, experience):
        return self._callmethod('store', (experience,))

    def load(self):
        self._callmethod('load', ())

    def save(self):
        self._callmethod('save', ())

    def sample(self, global_step):
        while True:
            experience, w, rank_e_id = self._callmethod('sample', (global_step,))
            if experience != False:
                break
            raise ValueError('get experience failed!')
        return experience[0], w[0], rank_e_id[0]

    def update_priority(self, indices, delta):
        assert len(indices) == len(delta), "lens mismatch {} vs {}".format(len(indices), len(delta))
        return self._callmethod('update_priority', (indices, delta,))


class Manager(BaseManager):
    pass


Manager.register('Experience', sync_expericence, ExperienceProxy)


class SaveStateHook(session_run_hook.SessionRunHook):
    def __init__(self, state_scope, reset_scope, meta_error, base_error, experience=None, keep_prob=0.9):
        assert isinstance(experience, ExperienceProxy)
        self._state_tensor = tf.global_variables(scope=state_scope)
        self._experience = experience
        self._first_run = True
        self._reset_op = tf.variables_initializer(tf.global_variables(scope=reset_scope))
        self._keep_prob = keep_prob
        self._global_step_tensor = tf.train.get_or_create_global_step()
        self._rank_e_id = 0
        self._meta_error = meta_error
        self._base_error = base_error
        self._state_norm = tf.global_norm(self._state_tensor)

    def before_run(self, run_context):  # pylint: disable=unused-argument
        if self._first_run:
            init_state = run_context.session.run(self._state_tensor)
            self._rank_e_id = self._experience.store(init_state)
            print('get init state, id: {}'.format(self._rank_e_id))
            self._first_run = False
            self._meta_error_history = []
            self._top_base_error = 0.0
            self._top_rank_e_id = self._rank_e_id
            self._top_state_norm = None
        self._should_reset = np.random.random() > self._keep_prob
        args = {'meta_error': self._meta_error}
        if self._should_reset:
            args["global_step"] = self._global_step_tensor
            args['base_error'] = self._base_error
            args['state_norm'] = self._state_norm
        return SessionRunArgs(args)

    def after_run(self, run_context, run_values):
        self._meta_error_history.append(run_values.results["meta_error"])
        if self._should_reset:
            base_error = run_values.results["base_error"]
            state_norm = run_values.results["state_norm"]
            if base_error <= self._top_base_error and \
                    (self._top_state_norm is None or
                     state_norm < self._top_state_norm * 1.2):
                print('store state based {}, base_error: {}, state_norm: {}'.format(self._rank_e_id, base_error,
                                                                                    state_norm))
                current_state, _ = run_context.session.run([self._state_tensor, self._reset_op])
                self._top_rank_e_id = self._experience.store(current_state)
                self._top_base_error = base_error
                self._top_state_norm = state_norm
            else:
                run_context.session.run(self._reset_op)
            delta = np.exp(np.mean(self._meta_error_history))
            self._experience.update_priority([self._rank_e_id], [delta])
            new_state, _, new_rank_e_id = self._experience.sample(run_values.results["global_step"])
            for var, val in zip(self._state_tensor, new_state):
                var.load(val, run_context.session)
            self._rank_e_id = new_rank_e_id
            self._meta_error_history = []
            print('reset with id: {}, top:{}'.format(self._rank_e_id, self._top_rank_e_id))


class RecordStateHook(session_run_hook.SessionRunHook):
    def __init__(self, state_scope, total_step, account, loss, experience=None):
        assert isinstance(experience, ExperienceProxy)
        self._state_tensor = tf.global_variables(scope=state_scope)
        self._experience = experience
        self._account = 0
        self._gap = total_step // account
        print('total_step: {}, gap: {}'.format(total_step, self._gap))
        self._loss = loss
        self._step = tf.train.get_or_create_global_step()

    def before_run(self, run_context):  # pylint: disable=unused-argument
        step = run_context.session.run(self._step)
        args = {}
        self._should_save = step % self._gap == 0
        if self._should_save:
            args['state'] = self._state_tensor
            args['loss'] = self._loss
        return SessionRunArgs(args)

    def after_run(self, run_context, run_values):
        if self._should_save:
            state = run_values.results["state"]
            loss = run_values.results["loss"]
            rank_e_id = self._experience.store(state)
            print('store state {}, loss {}'.format(rank_e_id, loss))
