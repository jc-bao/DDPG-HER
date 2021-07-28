import numpy as np
import gym
import gym_xarm
from ddpg_agent import DDPG_Agent
from utils import get_args

'''
TODO:
1. add cuda
2. add mpi
3. action normalizer
4. add ramdoness and gaussian noise to action 
'''

if __name__ == '__main__':
    args = get_args()
    env = gym.make(args.env_name)
    env.seed(args.seed)
    ddpg_trainer = DDPG_Agent(args, env)
    ddpg_trainer.learn()


