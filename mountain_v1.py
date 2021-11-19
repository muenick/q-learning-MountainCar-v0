import numpy as np
import gym
from gym import wrappers
from time import time # just to have timestamps in the files



env = gym.make('MountainCar-v0')

#env = wrappers.RecordEpisodeStatistics(env)
env = wrappers.RecordVideo(env, './videosDumm/'+ str(time()) + '/')

env.reset()

done = False

while not done: 
    action = 2
    env.step(action)
    env.render()