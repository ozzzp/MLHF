#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 

import rank_based


def test():
    meta_learn_batch = 1
    conf = {'size': 200,
            'learn_start': 1,
            'partition_num': 10,
            'total_step': 100000,
            'batch_size': meta_learn_batch}
    '''
    conf = {'size': 50,
            'learn_start': 10,
            'partition_num': 5,
            'total_step': 100,
            'batch_size': 4}
    '''
    experience = rank_based.Experience(conf)

    # insert to experience
    print('test insert experience')
    for i in range(0, conf['learn_start']):
        # tuple, like(state_t, a, r, state_t_1, t)
        to_insert = (i, 1, 1, i, 1)
        experience.store(to_insert)
    print(experience.priority_queue)

    # sample
    print('test sample')
    sample, w, e_id = experience.sample(0)
    print(sample)
    print(w)
    print(e_id)

    # update delta to priority
    print('test update delta')
    delta = [v for v in range(1, len(e_id) + 1)]
    experience.update_priority(e_id, delta)
    print(experience.priority_queue)
    sample, w, e_id = experience.sample(51)
    print(sample)
    print(w)
    print(e_id)

    # rebalance
    print('test rebalance')
    experience.rebalance()
    print(experience.priority_queue)


def main():
    test()


if __name__ == '__main__':
    main()

