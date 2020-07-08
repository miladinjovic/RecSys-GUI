# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:39:26 2020

@author: Jovic
"""

from surprise import AlgoBase
from surprise import PredictionImpossible
from YahooDataset import YahooDataset
import math
import numpy as np
import heapq
import pandas as pd


class ContentUserCollaborativeAlgorithm(AlgoBase):
    nearestNeigbors = {}

    def __init__(self, k=40, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k
        yd = YahooDataset()
        self.scores =  yd.loadMovies()[3]
        
    def fit(self, trainset):
        self.trainset = trainset
        trainset.rating_scale = (1, 13)
        AlgoBase.fit(self, trainset)
        simsMatrix = self._compute_similarities()
        
        for userId in range(trainset.n_users):
            similarityRow = simsMatrix[userId]
            kNeighbors = heapq.nlargest(10,  [(innerId, score) for (innerId, score) in  enumerate(similarityRow) if innerId!=userId], key=lambda t: t[1])
            self.nearestNeigbors[userId] = kNeighbors
            
        print("...done.")
        
    def _computeSimilarity(self, movie1, movie2):
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(movie1)):
            x = movie1[i]
            y = movie2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        
        if sumxy == 0 :
            return 0
        
        return sumxy/math.sqrt(sumxx*sumyy)
     
        
    def _compute_similarities(self):
        trainset = self.trainset
        
        similarities = np.zeros((trainset.n_users, trainset.n_users))
        
        for thisUser in range(trainset.n_users):
            print(thisUser)
            if (thisUser % 100 == 0):
                print(thisUser, " of ", self.trainset.n_users)
                
            thisUserProfile = self._createUserProfile(thisUser)
        
            for otherUser in range(thisUser+1, trainset.n_users):
                print("OTHER USER", otherUser)
                otherUserProfile = self._createUserProfile(otherUser)
                
                similarity = self._computeSimilarity(thisUserProfile, otherUserProfile)
                similarities[thisUser, otherUser] = similarity
                similarities[otherUser, thisUser] = similarity
       
    
    def _createUserProfile(self, user):
        trainset = self.trainset
        userProfile = np.zeros(10)
        numItems = 0
        
        for movieId, rating in trainset.ur[user]:
            movieId = int(trainset.to_raw_iid(movieId))
            if movieId == 0:
                continue
            
            numItems = numItems + 1
            profile = np.multiply(self.scores.loc[movieId], rating)
            userProfile = np.add(userProfile, profile)
            
        return np.divide(userProfile, numItems)
        
               
    
    def estimate(self, u, i):
        
        if not self.trainset.knows_item(i):
            raise PredictionImpossible('Item is unknown.')
            
        
        thisMovieID = int(self.trainset.to_raw_iid(i))
        if (thisMovieID == 0):
            raise PredictionImpossible('Item has no content data.')

        neighbors = []
        for rating in self.trainset.ur[u]:
            otherMovieID = int(self.trainset.to_raw_iid(rating[0]))
            
            if(otherMovieID == 0):
                continue
            
            similarity  = self.computeSimilarity(self.scores.loc[thisMovieID], self.scores.loc[otherMovieID])
            
            neighbors.append((similarity, rating[1]))
        
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
        
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if (simScore > 0):
                simTotal += simScore
                weightedSum += simScore * rating
            
        if (simTotal == 0):
            raise PredictionImpossible('No neighbors')

        predictedRating = weightedSum / simTotal
        
        return predictedRating