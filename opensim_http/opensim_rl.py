#from gym.envs.base import Env
#from gym.envs.base import Step
from gym.spaces import Box
import numpy as np
from client_http import *
import random

from observation_processor import generate_observation as go

class OpenSimEnv(object):
	def __init__(self,visualize=False,difficulty=2,remote_base = 'http://10.42.0.1:5000',seed=None,skipcount=1):
		self.remote_base = remote_base
		self.stepcount = 0
		self.visualize = visualize
		self.skipcount = skipcount # 4
		self.difficulty = difficulty
		self.env = Client(remote_base)
		self.instance_id = self.env.env_create(self.difficulty,self.visualize)
		action_info = self.env.env_action_space_info(self.instance_id)
		obs_info = self.env.env_observation_space_info(self.instance_id)
		self._observation_space = Box(low=min(obs_info['low']), high=max(obs_info['high']), shape=(1*41,))
		print("observation space: {}".format(self._observation_space))
		self._action_space = Box(low=0.0, high=1.0, shape=(action_info['shape'][0],))
		print("action space: {}".format(self._action_space))
		self.old_observation = None
		self.metadata = {'render.modes': []}
		self.reward_range = (-np.inf, np.inf) 
		self.stepcount = 0
		self.episode = 0               
	@property
	def observation_space(self):
		return self._observation_space
	@property
	def action_space(self):
		return self._action_space
	@property
	def horizon(self):
		return 1000
	def recreate(self):
		self.env = Client(self.remote_base)
		self.instance_id = self.env.env_create(self.difficulty,self.visualize)
	def seed(self, seed=None):
		return
	def obg(self,plain_obs):
		# observation generator
		# derivatives of observations extracted here.
		processed_observation, self.old_observation = go(plain_obs, self.old_observation, step=self.stepcount)
		return np.array(processed_observation)

	def reset(self,difficulty=2):
		#self.difficulty = random.randint(0,3)
		self.episode +=1
		if self.episode % 20 == 0:
			self.exit()
			self.recreate()
		self.stepcount=0
		self.old_observation = None
		observation = self.env.env_reset(self.instance_id,difficulty = 2)
		o = self.obg(observation)
		return o
	def step(self, action):
		action = [float(i) for i in action]
		sr = 0
		for j in range(self.skipcount):
			self.stepcount+=1
			oo,r,d,i = self.env.env_step(self.instance_id,action)
			o = self.obg(oo)
			sr += r

			if d == True:
				break
		
		return o,sr,d,i
	def exit(self):
		self.env.env_close(self.instance_id)
		#del self.features
	def render(self):
		if self.visualize == False:
			print('current state:')
