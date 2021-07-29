import torch
import gym
import numpy as np
from utils import get_args
from models import Actor
import gym_xarm

if __name__ == '__main__':
    args = get_args()
    # 1. load model
    model_path = args.save_dir + args.env_name + '/model.pt'
    success_rate, o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
    # 2. make gym env
    env = gym.make(args.env_name)
    obs_out = env.reset()
    env_params = {
        'obs': obs_out['observation'].shape[0],
        'goal': obs_out['desired_goal'].shape[0],
        'action': env.action_space.shape[0],
        'action_max': env.action_space.high[0],
    }
    for i in success_rate:
        print(i)
    # 3. make actor network
    actor_network = Actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    for i in range(args.demo_length):
        obs_out = env.reset()
        obs = obs_out['observation']
        g = obs_out['desired_goal']
        for t in range(env._max_episode_steps):
            env.render()
            # clip
            obs_clip = np.clip(obs, -args.clip_obs, args.clip_obs)
            g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
            # normalize
            obs_norm = np.clip((obs_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
            g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
            # concatenate
            inputs = torch.tensor(np.concatenate([obs_norm, g_norm]), dtype=torch.float32)
            with torch.no_grad():
                actions = actor_network(inputs).detach().cpu().numpy().squeeze()
            obs_out, _, _, info = env.step(actions)
            obs= obs_out['observation']
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))