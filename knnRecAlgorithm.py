# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 15:54:04 2020

@author: Jovic
"""

from surprise import AlgoBase
from surprise import KNNBasic, KNNBaseline, KNNWithMeans
import heapq
from collections import defaultdict

class knnRecAlgorithm(AlgoBase):
    
    nearestNeigbors = {}
    
    def __init__(self, k, sim_options):
        AlgoBase.__init__(self)
        self.k = k
        self.sim_options = sim_options
        
    def fit(self, trainset):
        self.trainset = trainset
        self.trainset.rating_scale = (1, 13)
        AlgoBase.fit(self, trainset) 
        
        # sim_options = {'name': 'cosine',
        #        'user_based': True
        #        }
        
        model = KNNBasic(sim_options=self.sim_options, k = self.k)
        model.fit(trainset)
        simsMatrix = model.compute_similarities()
        
        
        for userId in range(trainset.n_users):
            
            similarityRow = simsMatrix[userId]
            kNeighbors = heapq.nlargest(10,  [(innerId, score) for (innerId, score) in  enumerate(similarityRow) if innerId!=userId], key=lambda t: t[1])
            self.nearestNeigbors[userId] = kNeighbors
            

        print("...done.")
                
        return self 
    
    def estimate(self, u, i):
        score = 0
        
        for otherUserId, similarity in self.nearestNeigbors[u]:
            for movieId, rating in self.trainset.ur[otherUserId]:
                if movieId == i:
                    score += (rating / 13) * similarity
                    break

        return score
    
#    def recommend(self, u):
#        
#        topN = []
#        candidates = defaultdict(float)
#        watched = [rating[0] for rating  in self.trainset.ur[u]]
#        
#        for otherUserId, similarity in self.nearestNeigbors[u]:
#            for movieId, rating in self.trainset.ur[otherUserId]:
#                if movieId not in watched:
#                    movieId = int(self.trainset.to_raw_iid(movieId))
#                    candidates[movieId] += (rating / 13) * similarity
#        
#                
#        topN = heapq.nlargest(10, candidates.items(), key=lambda t: t[1])
#        
#        return topN
        
        
                
    