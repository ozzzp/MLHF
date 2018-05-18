# Forked version: Prioritized Experience Replay (simplified and extended)

### Usage
    import proportional
    import rank_based

    if params['experience_type'] == 'rank':
        experience = rank_based.Experience(replay_params)
    elif params['experience_type'] == 'proportional':
        experience = proportional.Experience(replay_params)
        
    experience.store(state)
    batch_experience, batch_w, batch_rank_e_id = experience.sample(global_step=variables['frames_total'])
    experience.update_priority(batch_rank_e_id, td_errors)

### Rank-based
use binary heap tree as priority queue, and build an Experience class to store and retrieve the sample
  
    Interface:
    * All interfaces are in rank_based.py
    * init conf, please read Experience.__init__ for more detail, all parameters can be set by input conf
    * replay sample store: Experience.store
    	params: [in] experience, sample to store
    	returns: bools, True for success, False for failed
    * replay sample sample: Experience.sample
    	params: [in] global_step, used for cal beta
    	returns: 
    		experience, list of samples
    		w, list of weight
    		rank_e_id, list of experience's id, used for update priority value
    * update priority value: Experience.update
    	params: 
    		[in] indices, rank_e_ids
    		[in] delta, new TD-error

### Proportional

    Interface:
    * All interfaces are in proportional.py
    Use the same way as Rank-based

### Reference
1. "Prioritized Experience Replay" http://arxiv.org/abs/1511.05952
2. [Atari](https://github.com/Kaixhin/Atari) by @Kaixhin, Atari uses torch to implement rank-based algorithm.
