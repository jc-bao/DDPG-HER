import numpy as np

class HERSampler:
    def __init__(self, replay_strategy, replay_k, reward_func = None):
        self.reward_func = reward_func
        self.replay_k = replay_k
        if replay_strategy == 'future':
            self.future_p = 1 - (1./(1+replay_k))
        else:
            self.future_p = 0 

    def sample_her_transitions(self, batch, size):
        T = batch['actions'].shape[1]
        n_rollout = batch['actions'].shape[0]
        # 1. select
        idx_rollout = np.random.randint(0, n_rollout, size)
        t = np.random.randint(T, size=size)
        transitions = {
            key: batch[key][idx_rollout, t].copy() for key in batch.keys()
        }
        # 2. sample
        sample_idx = np.where(np.random.uniform(size=size)<self.future_p)
        offset = np.random.uniform(size=size) * (T - t) + 1
        offset = offset.astype(int)
        future_t = (t + offset)[sample_idx]
        future_ag = batch['ag'][idx_rollout[sample_idx], future_t]
        # 3. replace and update reward
        transitions['g'][sample_idx] = future_ag
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        # 4. reshape and return 
        transitions = {key: transitions[key].reshape(size, *transitions[key].shape[1:]) for key in transitions.keys()}
        return transitions

