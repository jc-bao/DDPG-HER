import numpy as np
import gym
import gym_xarm
from ddpg_agent import DDPG_Agent
from utils import get_args
from mpi4py import MPI
import random
import torch
import os

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
    # set random seeds for reproduce
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # training
    ddpg_trainer = DDPG_Agent(args, env)
    ddpg_trainer.learn()