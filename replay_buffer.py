import numpy as np

class ReplayBuffer:
    def __init__(self, env_params, buffer_size, sample_func):
        self.T = env_params['max_timesteps']
        self.env_params = env_params
        self.size = buffer_size//self.T
        self.sample_func = sample_func
        self.current_size = 0
        self.n_transitions_stored = 0 # total transitions
        self.buffers = {
            'obs': np.empty([buffer_size, self.T+1, self.env_params['obs']]),
            'ag': np.empty([buffer_size, self.T+1, self.env_params['goal']]),
            'g': np.empty([buffer_size, self.T, self.env_params['goal']]),
            'actions': np.empty([buffer_size, self.T, self.env_params['action']])
        }

    def sample(self, batch_size):
        buffer = {
            key: self.buffers[key][:self.current_size] for key in self.buffers.keys()
        }
        buffer['obs_next'] = buffer['obs'][:,1:,:]
        buffer['ag_next'] = buffer['ag'][:,1:,:]
        return self.sample_func(buffer, batch_size)
    
    def store_episode(self, cyc_obs, cyc_ag, cyc_g, cyc_actions):
        batch_size = cyc_obs.shape[0]
        if self.current_size+ batch_size <= self.size:
            idx = np.arange(self.current_size, self.current_size+  batch_size)
        elif self.current_size < self.size:
            overflow =   batch_size - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size,   batch_size)
        self.current_size = min(self.size, self.current_size+batch_size)
        if batch_size == 1:
            idx = idx[0]
        self.buffers['obs'][idx] = cyc_obs
        self.buffers['ag'][idx] = cyc_ag
        self.buffers['g'][idx] = cyc_g
        self.buffers['actions'][idx] = cyc_actions
        self.n_transitions_stored += self.T * batch_size