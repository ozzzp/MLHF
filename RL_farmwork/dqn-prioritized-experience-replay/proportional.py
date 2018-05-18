#! -*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
#import random
import sum_tree


class BetaSchedule(object):
    def __init__(self, conf=None):
        self.batch_size = int(conf['batch_size'] if 'batch_size' in conf else 32)

        self.beta_zero = conf['beta_zero'] if 'beta_zero' in conf else 0.5
        self.learn_start = int(conf['learn_start'] if 'learn_start' in conf else 1000)
        # http://www.evernote.com/l/ACnDUVK3ShVEO7fDm38joUGNhDik3fFaB5o/
        self.total_steps = int(conf['total_steps'] if 'total_steps' in conf else 100000)
        self.beta_grad = (1 - self.beta_zero) / (self.total_steps - self.learn_start)

    def get_beta(self, global_step):
        # beta, increase by global_step, max 1
        beta = min(self.beta_zero + (global_step - self.learn_start - 1) * self.beta_grad, 1)
        return beta, self.batch_size



class Experience(object):
    """ The class represents prioritized experience replay buffer.

    The class has functions: store samples, pick samples with 
    probability in proportion to sample's priority, update 
    each sample's priority, reset alpha.

    see https://arxiv.org/pdf/1511.05952.pdf .

    """
    
    def __init__(self, conf = {}):
        """ Prioritized experience replay buffer initialization.
        
        Parameters
        ----------
        memory_size : int
            sample size to be stored
        batch_size : int
            batch size to be selected by `select` method
        alpha: float
            exponent determine how much prioritization.
            Prob_i \sim priority_i**alpha/sum(priority**alpha)
        """
        self.beta_sched = BetaSchedule(conf)
        self._max_priority = 1.0

        self.index = 0
        self.record_size = 0
        self.isFull = False
        self.prioritized_replay_eps = 1e-6

        if not conf is None:
            memory_size = int(conf['size']) if 'size' in conf else 10000
            self.memory_size = memory_size
            self.tree = sum_tree.SumTree(memory_size)
            #self.batch_size = batch_size
            self.alpha = conf['alpha'] if 'alpha' in conf else 0.7
            if 'prioritized_replay_eps' in conf:
                self.prioritized_replay_eps = float(conf['prioritized_replay_eps'])

    def save(self, filename):
        # data = np.array([
        #     self.size,
        #     self.replace_flag,
        #     self.alpha,
        #     self.beta_sched.beta_zero,
        #     self.beta_sched.batch_size,
        #     self.beta_sched.learn_start,
        #     self.beta_sched.total_steps,
        #     self.index,
        #     self.record_size,
        #     self.isFull,
        #     self._experience,
        #     self.priority_queue.priority_queue,
        #     self.priority_queue.p2e,
        #     self.priority_queue.e2p,
        #     self.priority_queue.size
        # ])
        # np.save(filename, data)
        assert False, "proportional.experience.save() is not implemented!"
        pass

    def load(self, filename):
        # data = np.load(filename)
        # self.size, \
        # self.replace_flag, \
        # self.alpha, \
        # self.beta_sched.beta_zero, \
        # self.beta_sched.batch_size, \
        # self.beta_sched.learn_start, \
        # self.beta_sched.total_steps, \
        # self.index, \
        # self.record_size, \
        # self.isFull, \
        # self._experience, \
        # self.priority_queue.priority_queue, \
        # self.priority_queue.p2e,\
        # self.priority_queue.e2p, \
        # self.priority_queue.size = data
        # self.priority_queue.balance_tree()
        assert False, "proportional.experience.load() is not implemented!"
        pass

    def fix_index(self):
        """
        get next insert index
        :return: index, int
        """
        if self.record_size <= self.memory_size:
            self.record_size += 1
        if self.index % self.memory_size == 0:
            self.index = 1
            return self.index
        else:
            self.index += 1
            return self.index

    def store(self, data, priority=None):
        """ Add new sample.
        
        Parameters
        ----------
        data : object
            new sample
        priority : float
            sample's priority
        """
        if priority is None:
            priority = self._max_priority
        self.fix_index()
        self.tree.add(data, priority**self.alpha)

    def sample(self, global_step, batch_size=None):
        beta, batch_size = self.beta_sched.get_beta(global_step)
        return self.select(beta, batch_size=batch_size)

    def select(self, beta, batch_size=32):
        """ The method return samples randomly.
        
        Parameters
        ----------
        beta : float
        
        Returns
        -------
        out : 
            list of samples
        weights: 
            list of weight
        indices:
            list of sample indices
            The indices indicate sample positions in a sum tree.
        """
        
        if self.tree.filled_size() < batch_size:
            return None, None, None

        out = []
        indices = []
        weights = []
        priorities = []
        rand_vals = np.random.rand(batch_size)
        for r in rand_vals:  #range(batch_size):
            #r = random.random()
            data, priority, index = self.tree.find(r)
            priorities.append(priority)
            weights.append((1./self.memory_size/priority)**beta if priority > 1e-16 else 0)
            indices.append(index)
            out.append(data)
            #self.update_priority([index], [0]) # To avoid duplicating
            
        
        self.update_priority(indices, priorities) # Revert priorities
        # weights /= max(weights) # Normalize for stability
        w = np.array(weights)
        w = np.divide(w,max(w))

        return out, w, indices

    def update_priority(self, indices, priorities):
        """ The methods update samples's priority.
        
        Parameters
        ----------
        indices : 
            list of sample indices
        """
        for i, p in zip(indices, priorities):
            self.tree.val_update(i, (p+self.prioritized_replay_eps)**self.alpha)
    
    def reset_alpha(self, alpha):
        """ Reset a exponent alpha.

        Parameters
        ----------
        alpha : float
        """
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [self.tree.get_val(i)**-old_alpha for i in range(self.tree.filled_size())]
        self.update_priority(range(self.tree.filled_size()), priorities)

    def rebalance(self):
        pass
            
        
        
