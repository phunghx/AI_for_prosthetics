#!/usr/bin/env python

import opensim as osim

from osim.redis.client import Client
from osim.env import *
import numpy as np
import argparse
import os


import gym
from gym import wrappers
from gym.spaces import Discrete, Box

from math import *
import random
import time

#from winfrey import wavegraph

from noise import one_fsq_noise

import tensorflow as tf
import canton as ct
from canton import *

"""
NOTE: For testing your submission scripts, you first need to ensure 
that redis-server is running in the background
and you can locally run the grading service by running this script : 
https://github.com/crowdAI/osim-rl/blob/master/osim/redis/service.py
The client and the grading service communicate with each other by 
pointing to the same redis server.
"""

"""
Please ensure that `visualize=False`, else there might be unexpected errors 
in your submission
"""

features = sorted(['joint_pos','joint_vel','joint_acc','body_pos','body_vel','body_acc',
            'body_pos_rot','body_vel_rot','body_acc_rot','misc'])
body_parts = sorted(['femur_r','pros_tibia_r','pros_foot_r','femur_l','tibia_l','talus_l','calcn_l','toes_l','torso','head'])
misc = sorted(['mass_center_pos','mass_center_vel','mass_center_acc'])

def obg(plain_obs):
                plain_obs[plain_obs > 1000] = 0
                plain_obs[plain_obs < -1000] = 0
                return plain_obs

def concatenate_state(state):
    result = []
    #fout = open('state.txt','w')
    #for k,v in state.items():
    #   fout.write(str(k) + ' >>> ' + str(v) + '\n')
    #fout.close()
    #state['misc']['mass_center_pos'][0]=0.1
    #state['body_pos']['pelvis'][0] = 0.0
    parts = sorted(state.keys())
    pelvis_s = {}
    for part in parts:
      if part[:4] != 'body':
        
        if type(state[part]) == dict:
          sub_parts = sorted(state[part].keys())
          for sub_part in sub_parts:
            if type(state[part][sub_part]) == dict:
               typek = sorted(state[part][sub_part].keys())
               for k in typek:
                   result.append(state[part][sub_part][k])
            else:
               result = result + state[part][sub_part].copy()
        else:
           result = result + state[part].copy()
      else:
        pelvis = state[part]['pelvis']
        pelvis_add = pelvis.copy()
        result = result + pelvis_add.copy()
        for sub_part in body_parts:
            result = result + (np.array(state[part][sub_part]) - np.array(pelvis)).tolist()     
    result =  np.array(result)
    return obg(result)



class nnagent(object):
    def __init__(self,
    observation_space_dims,
    action_space,
    stack_factor=1,
    discount_factor=.99, # gamma
    # train_skip_every=1,
    train_multiplier=1,
    ):
        self.global_step = 0
        self.batch_size= 128
        self.memSize = 1000000
        self.totalStep = 10000000
        self.render = True
        self.training = True
        self.noise_source = one_fsq_noise()
        self.train_counter = 0
        # self.train_skip_every = train_skip_every
        self.train_multiplier = train_multiplier
        self.observation_stack_factor = stack_factor
        self.frequence = 4
        self.inputdims = observation_space_dims * self.observation_stack_factor
        # assume observation_space is continuous

        self.is_continuous = True if isinstance(action_space,Box) else False

        if self.is_continuous: # if action space is continuous

            low = action_space.low
            high = action_space.high

            num_of_actions = action_space.shape[0]

            self.action_bias = high/2. + low/2.
            self.action_multiplier = high - self.action_bias

            # say high,low -> [2,7], then bias -> 4.5
            # mult = 2.5. then [-1,1] multiplies 2.5 + bias 4.5 -> [2,7]

            def clamper(actions):
                return np.clip(actions,a_max=action_space.high,a_min=action_space.low)

            self.clamper = clamper
        else:
            num_of_actions = action_space.n

            self.action_bias = .5
            self.action_multiplier = .5 # map (-1,1) into (0,1)

            def clamper(actions):
                return np.clip(actions,a_max=1.,a_min=0.)

            self.clamper = clamper

        self.outputdims = num_of_actions
        self.discount_factor = discount_factor
        ids,ods = self.inputdims,self.outputdims
        print('inputdims:{}, outputdims:{}'.format(ids,ods))

        self.actor = self.create_actor_network(ids,ods)
        self.critic = self.create_critic_network(ids,ods)
        self.actor_target = self.create_actor_network(ids,ods)
        self.critic_target = self.create_critic_network(ids,ods)

        # print(self.actor.get_weights())
        # print(self.critic.get_weights())

        self.feed,self.joint_inference,sync_target = self.train_step_gen()

        sess = ct.get_session()
        sess.run(tf.global_variables_initializer())

        sync_target()

        import threading as th
        self.lock = th.Lock()
        '''
        if not hasattr(self,'wavegraph'):
            num_waves = self.outputdims*2+1
            def rn():
                r = np.random.uniform()
                return 0.2+r*0.4
            colors = []
            for i in range(num_waves-1):
                color = [rn(),rn(),rn()]
                colors.append(color)
            colors.append([0.2,0.5,0.9])
            self.wavegraph = wavegraph(num_waves,'actions/noises/Q',np.array(colors))
         '''
    # the part of network that the input and output shares architechture
    def create_common_network(self,inputdims,outputdims):
        # timesteps = 8
        # dim_per_ts = int(inputdims/timesteps)
        # rect = Act('relu')
        # c = Can()
        #
        # if not hasattr(self,'gru'):
        #     # share parameters between actor and critic
        #     self.common_gru = GRU(dim_per_ts,128)
        #
        # gru = c.add(self.common_gru)
        #
        # # d1 = c.add(Dense(128,128))
        # d2 = c.add(Dense(128,outputdims))
        #
        # def call(i):
        #     # shape i: [Batch Dim*Timesteps]
        #
        #     batchsize = tf.shape(i)[0]
        #
        #     reshaped = tf.reshape(i,[batchsize,timesteps,dim_per_ts])
        #     # [Batch Timesteps Dim]
        #
        #     o = gru(reshaped)
        #     # [Batch Timesteps Dim]
        #
        #     ending = o[:,timesteps-1,:]
        #
        #     # l1 = rect(d1(ending))
        #     l2 = rect(d2(ending))
        #     return l2
        #
        # c.set_function(call)
        # return c

        c = Can()
        rect = Act('lrelu',alpha=0.2)
        magic = 1/(0.5 + 0.5*0.2)
        # rect = Act('elu')
        d1 = c.add(Dense(inputdims,512,stddev=magic))
        d1_n = c.add(LayerNorm(512))
        d1a = c.add(Dense(512,256,stddev=magic))
        d1a_n = c.add(LayerNorm(256))
        d2 = c.add(Dense(256,outputdims,stddev=magic))
        d2_n = c.add(LayerNorm(outputdims))

        def call(i):
            # i = Lambda(lambda x:x/3)(i) # downscale
            i = rect(d1_n(d1(i)))
            
            i = rect(d1a_n(d1a(i)))
            i = rect(d2_n(d2(i)))
            # l2 = Lambda(lambda x:x/8)(l2) # downscale a bit
            return i
        c.set_function(call)
        return c

    # a = actor(s) : predict actions given state
    def create_actor_network(self,inputdims,outputdims):
        # add gaussian noise.

        rect = Act('relu',alpha=0.2)
        magic = 1/(0.5 + 0.5*0.2)
        # rect = Act('elu')

        c = Can()
        c.add(self.create_common_network(inputdims,128))
        c.add(Dense(128,128))
        c.add(LayerNorm(128))
        c.add(rect)
        c.add(Dense(128,outputdims,stddev=1))

        if self.is_continuous:
            c.add(Act('tanh'))
            c.add(Lambda(lambda x: x*self.action_multiplier + self.action_bias))
        else:
            c.add(Act('softmax'))

        c.chain()
        return c

    # q = critic(s,a) : predict q given state and action
    def create_critic_network(self,inputdims,actiondims):
        rect = Act('lrelu',alpha=0.2)
        magic = 1/(0.5 + 0.5*0.2)
        # rect = Act('elu')

        c = Can()
        concat = Lambda(lambda x:tf.concat(x,axis=1))

        # concat state and action
        den0 = c.add(self.create_common_network(inputdims,128))
        # den1 = c.add(Dense(256, 256))
        den2 = c.add(Dense(128+actiondims, 128,stddev=magic))
        den2_n = c.add(LayerNorm(128))
        den3 = c.add(Dense(128,48,stddev=magic))
        den3_n = c.add(LayerNorm(48))
        den4 = c.add(Dense(48,1,stddev=1))

        def call(i):
            state = i[0]
            action = i[1]
            i = den0(state)

            i = concat([i,action])
            i = rect(den2_n(den2(i)))
            i = rect(den3_n(den3(i)))
            i = den4(i)

            q = i
            return q
        c.set_function(call)
        return c

    def train_step_gen(self):
        s1 = tf.placeholder(tf.float32,shape=[None,self.inputdims])
        a1 = tf.placeholder(tf.float32,shape=[None,self.outputdims])
        r1 = tf.placeholder(tf.float32,shape=[None,1])
        isdone = tf.placeholder(tf.float32,shape=[None,1])
        wWeight = tf.placeholder(tf.float32,shape=[None,1]) #rank replay buffer
        s2 = tf.placeholder(tf.float32,shape=[None,self.inputdims])

        # 1. update the critic
        a2 = self.actor_target(s2)
        q2 = self.critic_target([s2,a2])
        q1_target = r1 + (1-isdone) * self.discount_factor * q2
        q1_predict = self.critic([s1,a1])
        td_error = (q1_target - q1_predict)
        critic_loss = tf.reduce_mean(tf.multiply((q1_target - q1_predict)**2,wWeight))
        # produce better prediction

        # 2. update the actor
        a1_predict = self.actor(s1)
        q1_predict = self.critic([s1,a1_predict])
        actor_loss = tf.reduce_mean(- q1_predict)
        # maximize q1_predict -> better actor

        # 3. shift the weights (aka target network)
        tau = tf.Variable(1e-3) # original paper: 1e-3. need more stabilization
        aw = self.actor.get_weights()
        atw = self.actor_target.get_weights()
        cw = self.critic.get_weights()
        ctw = self.critic_target.get_weights()
        #import pdb;pdb.set_trace()
        one_m_tau = 1-tau

        shift1 = [tf.assign(atw[i], aw[i]*tau + atw[i]*(one_m_tau))
            for i,_ in enumerate(aw)]
        shift2 = [tf.assign(ctw[i], cw[i]*tau + ctw[i]*(one_m_tau))
            for i,_ in enumerate(cw)]

        # 4. inference
        
        set_training_state(False)
        a_infer = self.actor(s1)
        q_infer = self.critic([s1,a_infer])
        set_training_state(True)

        # 5. L2 weight decay on critic
        decay_c = tf.reduce_sum([tf.reduce_sum(w**2) for w in cw])* 1e-7
        decay_a = tf.reduce_sum([tf.reduce_sum(w**2) for w in aw])* 1e-7

        decay_c = 0
        decay_a = 0

        # optimizer on
        # actor is harder to stabilize...
        opt_actor = tf.train.AdamOptimizer(1e-4)
        opt_critic = tf.train.AdamOptimizer(3e-4)
        # opt_actor = tf.train.RMSPropOptimizer(1e-3)
        # opt_critic = tf.train.RMSPropOptimizer(1e-3)
        cstep = opt_critic.minimize(critic_loss+decay_c, var_list=cw)
        astep = opt_actor.minimize(actor_loss+decay_a, var_list=aw)

        self.feedcounter=0
        def feed(memory,wWeightF,e_id):
            [s1d,a1d,r1d,isdoned,s2d] = memory # d suffix means data
            sess = ct.get_session()
            res = sess.run([critic_loss,actor_loss,
                cstep,astep,shift1,shift2,td_error],
                feed_dict={
                s1:s1d,a1:a1d,r1:r1d,isdone:isdoned,s2:s2d,tau:5e-4,wWeight:wWeightF
                })
            self.rpm.update_priority(e_id, res[-1])

            #debug purposes
            self.feedcounter+=1
            #if self.feedcounter%10==0:
            #    print(' '*30, 'closs: {:6.4f} aloss: {:6.4f}'.format(
            #        res[0],res[1]),end='\r')

            # return res[0],res[1] # closs, aloss

        def joint_inference(state):
            sess = ct.get_session()
            res = sess.run([a_infer,q_infer],feed_dict={s1:state})
            return res

        def sync_target():
            sess = ct.get_session()
            sess.run([shift1,shift2],feed_dict={tau:1.})

        return feed,joint_inference,sync_target

    def train(self,verbose=1,current_step=1):
        memory = self.rpm
        batch_size = self.batch_size
        total_size = batch_size
        epochs = 1

        # self.lock.acquire()
        if memory.size() > total_size * 128:
            #print("Training ...")
            #if enough samples in memory
            for i in range(self.train_multiplier):
                # sample randomly a minibatch from memory
                [s1,a1,r1,isdone,s2],wWeight,e_id = memory.sample_batch(current_step)
                # print(s1.shape,a1.shape,r1.shape,isdone.shape,s2.shape)
                
                self.feed([s1,a1,r1,isdone,s2],wWeight,e_id)

        # self.lock.release()

    def feed_one(self,tup):
        self.rpm.add(tup)


    # one step of action, given observation
    def act(self,observation,curr_noise=None):
        actor,critic = self.actor,self.critic
        obs = np.reshape(observation,(1,len(observation)))

        # actions = actor.infer(obs)
        # q = critic.infer([obs,actions])[0]
        # self.lock.acquire()
        [actions,q] = self.joint_inference(obs)
        # self.lock.release()

        actions,q = actions[0],q[0]

        if curr_noise is not None:
            disp_actions = (actions-self.action_bias) / self.action_multiplier
            disp_actions = disp_actions * 5 + np.arange(self.outputdims) * 12.0 + 30

            noise = curr_noise * 5 - np.arange(self.outputdims) * 12.0 - 30

            # self.lock.acquire()
            #self.loggraph(np.hstack([disp_actions,noise,q]))
            # self.lock.release()
            # temporarily disabled.
        return actions



    def load_weights(self, episode):
        networks = ['actor']
        for name in networks:
            network = getattr(self,name)
            network.load_weights('./models/ddpg_'+name+str(episode)+'.npz')



env = ProstheticsEnv(visualize=False)


from gym.spaces import Box

#from observation_processor import processed_dims
processed_dims = 415
ob_space = Box(-3.0, 3.0, (processed_dims,))
action_space = Box(0, 1.0, (19,))


#weights = [500999,585999,590999,523999,505999,518999]
weights = [585999,591999,590999]
num_agent = len(weights)
agents = []


for i in range(num_agent):
  agents.append(nnagent(
    processed_dims,
    action_space,
    discount_factor=.98,
    # .99 = 100 steps = 4 second lookahead
    # .985 = somewhere in between.
    # .98 = 50 steps = 2 second lookahead
    # .96 = 25 steps = 1 second lookahead
    stack_factor=1,
    train_multiplier=1,
    )
  )
for i in range(num_agent):
  agents[i].training=False
  agents[i].load_weights(weights[i])

"""
Define evaluator end point from Environment variables
The grader will pass these env variables when evaluating
"""
REMOTE_HOST = os.getenv("CROWDAI_EVALUATOR_HOST", "127.0.0.1")
REMOTE_PORT = os.getenv("CROWDAI_EVALUATOR_PORT", 6379)
client = Client(
    remote_host=REMOTE_HOST,
    remote_port=REMOTE_PORT
)

# Create environment
observation = client.env_create()
observation = concatenate_state(observation)
"""
The grader runs N simulations of at most 1000 steps each. 
We stop after the last one
A new simulation starts when `clinet.env_step` returns `done==True`
and all the simulations end when the subsequent `client.env_reset()` 
returns a False
"""
stepno=0
total_reward = 0
while True:
    action = agents[0].act(observation)    
    for i in range(num_agent-1):
      action = action + agents[i+1].act(observation)    

    action = (1.0/num_agent)*action

    [observation, reward, done, info] = client.env_step([float(round(a,10)) for a in action])
    observation = concatenate_state(observation)
    stepno+=1
    total_reward+=reward
    if done:
        print('step',stepno,'total reward',total_reward)
        observation = client.env_reset()
        if not observation:
            break
        observation = concatenate_state(observation)
        stepno=0
        total_reward = 0




client.submit()



