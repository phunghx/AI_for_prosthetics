# from collections import deque
import numpy as np
import random

import pickle
class CircularQueue(object):
    #replay memory
    def __init__(self,length,shape):
        self.length = length
        self.queue = {}
        self.reset = False
        for i in range(self.length):
              self.queue[i] = np.zeros(shape)

    def push(self, obj):
        if self.reset:
           for i in range(self.length):
              self.queue[i].fill(0.0)
           self.reset = False
        else:
           for i in range(self.length-1):
              self.queue[i] = self.queue[i+1].copy()
        self.queue[self.length-1] = obj.copy()

    def pushReset(self,obj):
        for i in range(self.length-1):
              self.queue[i] = self.queue[i+1].copy()
        self.queue[self.length-1] = obj.copy()
        self.reset = True

    def clear(self):
        for i in range(self.length):
              self.queue[i].fill(0.0)
        
    def readAll(self):
        output = np.zeros((self.length,self.queue[0].shape[0]))
        for i in range(self.length):
              output[i] = self.queue[i]
        return output.flatten()
    def __del__(self):
        for i in range(self.length):
              del self.queue[i]
     

