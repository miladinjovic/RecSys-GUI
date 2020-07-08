# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 14:17:34 2020

@author: Jovic
"""
from surprise import AlgoBase

class WeightedHybridAlgorithm(AlgoBase):

    def __init__(self, algorithm1, algorithm2, weights=[]):
        AlgoBase.__init__(self)
        self.algorithm1 = algorithm1
        self.algorithm2 = algorithm2
        self.weights = weights
        
    def fit(self, trainset):
        trainset.rating_scale = (1, 13)
        AlgoBase.fit(self, trainset) 
        
        self.algorithm1.fit(trainset)
        self.algorithm2.fit(trainset)

        print("...done.")
                
        return self 
    
    def estimate(self, u, i):
        score = self.algorithm1.estimate(u, i)*self.weights[0] + self.algorithm2.estimate(u, i)[0]*self.weights[1]          
        return score
        