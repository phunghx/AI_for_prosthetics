# from collections import deque
import numpy as np
import random

import pickle
import rank_based
# replay buffer per http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
# for a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count = count + 1 
    delta = newValue - mean
    mean = mean + delta / count
    delta2 = newValue - mean
    M2 = M2 + delta * delta2

    return (count, mean, M2)

# retrieve the mean, variance and sample variance from an aggregate
def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    (mean, variance, sampleVariance) = (mean, M2/count, M2/(count - 1)) 
    if count < 2:
        return float('nan')
    else:
        return (mean, variance, sampleVariance)


class rpm(object):
    #replay memory
    def __init__(self,buffer_size,batch_size,nsteps):
        conf = {'size': buffer_size,
            'learn_start': 1000,
            'partition_num': 100,
            'total_step': nsteps,
            'batch_size': batch_size}
        self.experience = rank_based.Experience(conf)
        self.first = True
        #self.buffer_size = buffer_size
        #self.buffer = []

    def add(self, obj):
        '''
        while self.size() >= self.buffer_size:
            # self.buffer.popleft()
            # self.buffer = self.buffer[1:]
            self.buffer.pop(0)
        self.buffer.append(obj)
        '''
        
        #if self.first:
        #    self.existingAggregate = (1,obj[0],0)
        #    self.first = False
        #else:
        #    self.existingAggregate = update(self.existingAggregate, obj[0])
        self.experience.store(obj)

    def size(self):
        return len(self.experience._experience)
    def get_data_mean(self):
        (mean, variance, sampleVariance) = finalize(self.existingAggregate)
        return mean, sampleVariance
    def sample_batch(self,global_step):
        '''
        batch_size specifies the number of experiences to add
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least
        batch_size elements before beginning to sample from it.
        '''
        '''
        if self.size() < batch_size:
            batch = random.sample(self.buffer, self.size())
        else:
            batch = random.sample(self.buffer, batch_size)

        item_count = len(batch[0])
        res = []
        for i in range(item_count):
            # k = np.array([item[i] for item in batch])
            k = np.stack((item[i] for item in batch),axis=0)
            # if len(k.shape)==1: k = k.reshape(k.shape+(1,))
            if len(k.shape)==1: k.shape+=(1,)
            res.append(k)
        return res
        '''
        
        sample, w, e_id = self.experience.sample(global_step)
        #(mean, variance, sampleVariance) = finalize(self.existingAggregate)
        

        item_count = len(sample[0])
        res = []
        for i in range(item_count):
            # k = np.array([item[i] for item in batch])
            k = np.stack((item[i] for item in sample),axis=0)
            # if len(k.shape)==1: k = k.reshape(k.shape+(1,))
            if len(k.shape)==1: k.shape+=(1,)
            res.append(k)
        #res[0] = (res[0] - mean)/(np.sqrt(sampleVariance)+1e-7)
        #res[-1] = (res[-1] - mean)/(np.sqrt(sampleVariance)+1e-7)
        return res,w,e_id
    def update_priority(self, e_id, delta):
        self.experience.update_priority(e_id, delta)

    def rebalance(self):
        self.experience.rebalance()
    def save(self, pathname):
        pickle.dump(self.experience, open(pathname, 'wb'))
        print('memory dumped into',pathname)
    def load(self, pathname):
        self.experience = pickle.load(open(pathname, 'rb'))
        print('memory loaded from',pathname)
