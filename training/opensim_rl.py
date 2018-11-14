from gym import Env
#from rllab.envs.base import Step
from gym.spaces import Box
import numpy as np
from client_http import *
import random
class OpenSimEnv(Env):
	def __init__(self,visualize=False,difficulty=1,remote_base = 'http://10.42.0.1:5000',seed=None,skipcount=1):
		remote_base = remote_base
		self.stepcount = 0
		self.visualize = visualize
		self.skipcount = skipcount # 4
		self.difficulty = difficulty
		self.env = Client(remote_base)
		self.instance_id = self.env.env_create(self.difficulty,self.visualize)
		#action_info = self.env.env_action_space_info(self.instance_id)
		#obs_info = self.env.env_observation_space_info(self.instance_id)
		self._observation_space = Box(low=-1.0, high=1.0, shape=(415,))
		print("observation space: {}".format(self._observation_space))
		self._action_space = Box(low=0.0, high=1.0, shape=(19,))
		print("action space: {}".format(self._action_space))
		self.old_observation = None
	@property
	def observation_space(self):
		return self._observation_space
	@property
	def action_space(self):
		return self._action_space
	@property
	def horizon(self):
		return 50000
	def obg(self,plain_obs):
                plain_obs[plain_obs > 1000] = 0
                plain_obs[plain_obs < -1000] = 0
                return plain_obs
	def reset(self):

		observation = self.env.env_reset(self.instance_id,difficulty = 1)
		return self.obg(np.copy(observation))
	def step(self, action):
		action = [float(i) for i in action]
		sr = 0
		for j in range(self.skipcount):		
			[state, reward, terminal, info] = self.env.env_step(self.instance_id, action)
			self.stepcount+=1
			sr += reward
			if terminal==True:	break
		
		next_observation = self.obg(np.array(state))
		return next_observation, sr, terminal, info
	def exit(self):
		self.env.env_close(self.instance_id)
	def render(self):
		if self.visualize == False:
			print('current state:')
