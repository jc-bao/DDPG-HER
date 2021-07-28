import torch
import numpy as np
import os
from datetime import datetime
from mpi4py import mpi4py
from mpi_utils.mpi_utils import sync_networks, sync_grads

from models import Actor, Critic
from replay_buffer import ReplayBuffer
from her import HERSampler
from normalizer import Normalizer

from utils import get_env_paramters

from torch.utils.tensorboard import SummaryWriter

class DDPG_Agent:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.env_params = get_env_paramters(env)
        # tensorboard
        self.tb = SummaryWriter()
        # create A-C network
        self.actor_network = Actor(self.env_params)
        self.critic_network = Critic(self.env_params)
        # target network
        self.actor_target_network = Actor(self.env_params)
        self.critic_target_network = Critic(self.env_params)
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr = self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr = self.args.lr_critic)
        # HER sampler
        self.her_module = HERSampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # replay ruffer
        self.buffer = ReplayBuffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # normalizer
        self.o_norm = Normalizer(size = self.env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = Normalizer(size = self.env_params['goal'], default_clip_range=self.args.clip_range)
        # create dict for store
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

    def learn(self):
        '''
        train the network
        '''
        for epoch in range(self.args.n_epochs): # for evaluation
            # 1. collect experiences
            for _ in range(self.args.n_cycles): # for update network
                # 1. collect trajectory
                cyc_obs, cyc_ag, cyc_g, cyc_actions = [],[],[],[]
                for _ in range(self.args.num_rollouts_per_mpi): # for collect traj
                    ro_obs, ro_ag, ro_g, ro_actions = [],[],[],[]
                    obs_out = self.env.reset()
                    obs, ag, g = obs_out['observation'],obs_out['achieved_goal'],obs_out['desired_goal']
                    # collect rollout sample
                    for _ in range(self.env_params['max_timesteps']): # for explore 1 time
                        # 1. get action and save it
                        with torch.no_grad():
                            inputs = self._og2input(obs,g)
                            action = self.actor_network(inputs)
                            action = self._select_actions(action)
                        ro_actions.append(action.copy())
                        ro_obs.append(obs)
                        ro_ag.append(ag)
                        ro_g.append(g)
                        # 2. get new obs,reward
                        obs_out, _, _, _ = self.env.step(action)
                        obs, ag, g = obs_out['observation'],obs_out['achieved_goal'],obs_out['desired_goal']
                    ro_obs.append(obs)
                    ro_ag.append(ag)
                    cyc_obs.append(ro_obs)
                    cyc_ag.append(ro_ag)
                    cyc_g.append(ro_g)
                    cyc_actions.append(ro_actions)
                # 2. save episode
                cyc_obs = np.array(cyc_obs)
                cyc_ag = np.array(cyc_ag)
                cyc_g = np.array(cyc_g)
                cyc_actions = np.array(cyc_actions)
                self.buffer.store_episode(cyc_obs, cyc_ag, cyc_g, cyc_actions)
                self._update_normalizer(cyc_obs, cyc_ag, cyc_g, cyc_actions)
                # 3. update A-C (target) networks
                self._update_network()
            # 2. evaluate result and save model
            success_rate = self._eval_agent()
            if MPI.COMM_WORLD.Get_rank() == 0:
                self.tb.add_scalar("Success Rate", success_rate, epoch)
                print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, success_rate))
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], self.model_path + '/model.pt')

    def _update_normalizer(self, cyc_obs, cyc_ag, cyc_g, cyc_actions):
        T = cyc_actions.shape[1]
        batch = {'obs': cyc_obs, 
                       'ag': cyc_ag,
                       'g': cyc_g, 
                       'actions': cyc_actions, 
                       'obs_next': cyc_obs[:,1:,:],
                       'ag_next': cyc_ag[:,1:,:],
                       }
        transitions = self.her_module.sample_her_transitions(batch, T)
        o, g = self._clip_og(transitions['obs'], transitions['g'])
        self.o_norm.update(o)
        self.g_norm.update(g)

    def _select_actions(self, a):
        '''
        add noise and randomness to actions
        '''
        action_max = self.env_params['action_max']
        action = a.cpu().numpy().squeeze()
        # add noise
        action += self.args.noise_eps * action_max * np.random.randn(*action.shape)
        action = np.clip(action, -action_max, action_max)
        # random action
        random_action = np.random.uniform(low=-action_max, high=action_max, size=self.env_params['action'])
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_action - action)
        return action

    def _clip_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    def _og2input(self, o, g):
        # clip
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        # normalize
        o = self.o_norm.normalize(o)
        g = self.g_norm.normalize(g)
        # concatenate
        return torch.tensor(np.concatenate((o,g), axis=(o.ndim-1)),dtype=torch.float32)

    def _update_network(self):
        for _ in range(self.args.n_batches):
            # 1. sample buffer and clip
            transitions = self.buffer.sample(self.args.batch_size)
            # 2. preprocess: clip->normalize->tensor
            inputs_tensor = self._og2input(transitions['obs'], transitions['g'])
            inputs_next_tensor = self._og2input(transitions['obs_next'], transitions['g'])
            actions_tensor = torch.tensor(transitions['actions'], dtype = torch.float32)
            r_tensor = torch.tensor(transitions['r'], dtype = torch.float32)
            # 3. get Q-value -> Q Loss
            with torch.no_grad(): # target-Q
                actions_next = self.actor_target_network(inputs_next_tensor)
                q_next_value = self.critic_target_network(inputs_next_tensor, actions_next).detach()
                target_q_value = r_tensor + self.args.gamma * q_next_value
            q_value = self.critic_network(inputs_tensor, actions_tensor)
            critic_loss = (target_q_value - q_value).pow(2).mean()
            # 4. Actor Loss
            actions = self.actor_network(inputs_tensor)
            actor_loss = - self.critic_network(inputs_tensor, actions).mean() + self.args.action_l2 * (actions / self.env_params['action_max']).pow(2).mean()
            # 5. update A-C network
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            # 6. update target network
            self._update_target_network(self.actor_target_network, self.actor_network)
            self._update_target_network(self.critic_target_network, self.critic_network)
        
    def _update_target_network(self, target, source):
        p = self.args.polyak
        for t, s in zip(target.parameters(),source.parameters()):
            t.data.copy_((1-p)*s.data + p*t.data )

    def _eval_agent(self):
        success_list = []
        for _ in range(self.args.n_test_rollouts):
            obs_out = self.env.reset()
            obs, _, g = obs_out['observation'],obs_out['achieved_goal'],obs_out['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    inputs = self._og2input(obs,g)
                    actions = self.actor_network(inputs).detach().cpu().numpy().squeeze()
                obs_out, _, _, info = self.env.step(actions)
                obs, _, g = obs_out['observation'],obs_out['achieved_goal'],obs_out['desired_goal']
                success_list.append(info['is_success'])
        success_rate = sum(success_list)/len(success_list)
        global_success_rate = MPI.COMM_WORLD.allreduce(success_rate, op=MPI.SUM)
        return global_success_rate