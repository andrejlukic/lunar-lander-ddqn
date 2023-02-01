import gym
import random
import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def make_env():
    env = gym.make('CartPole-v0')
    states = env.observation_space.shape[0]
    actions = env.action_space.n
    print(actions)

if __name__ == "__main__":
    make_env()