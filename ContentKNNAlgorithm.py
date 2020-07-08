# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:59:19 2019

@author: Jovic
"""

from surprise import AlgoBase
from surprise import PredictionImpossible
from YahooDataset import YahooDataset
import math
import numpy as np
import heapq
import pandas as pd

class ContentKNNAlgorithm(AlgoBase):

    def __init__(self, k=40, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k
        yd = YahooDataset()
        _, self.genres, self.actors, self.scores =  yd.loadMovies()
    
    def createMovieGenreProfile(self, movieId):
         maxGenreID = self.genres[-1]
         
         genreIDList = self.genres[movieId]
         genreProfile = [0] * maxGenreID
         
         for genreID in genreIDList:
             genreProfile[genreID] = 1
        
         return genreProfile
    
    def createMovieActorProfile(self, movieId):
        maxActorID = self.actors[-1]
        
        actorIDList = self.actors[movieId]
        actorProfile = [0] * maxActorID
        
        for actorID in actorIDList:
            actorProfile[actorID] = 1
        
        return actorProfile
    
    def fit(self, trainset):
        trainset.rating_scale = (1, 13)
        AlgoBase.fit(self, trainset)
        
#        print("Computing content-based similarity matrix...")
#            
#        # Compute genre distance for every movie combination as a 2x2 matrix
#        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))
#        
#        for thisRating in range(self.trainset.n_items):
#            if (thisRating % 100 == 0):
#                print(thisRating, " of ", self.trainset.n_items)
#            
#            thisMovieID = int(self.trainset.to_raw_iid(thisRating))
#            thisMovieGenreProfile = self.createMovieGenreProfile(thisMovieID)
#            thisMovieActorProfile = self.createMovieActorProfile(thisMovieID)
#            
#            thisMovieProfile= thisMovieActorProfile + thisMovieGenreProfile
#        
#            for otherRating in range(thisRating+1, self.trainset.n_items):
#                otherMovieID = int(self.trainset.to_raw_iid(otherRating))
#                otherMovieGenreProfile = self.createMovieGenreProfile(otherMovieID)
#                otherMovieActorProfile = self.createMovieActorProfile(otherMovieID)
#                
#                otherMovieProfile=otherMovieActorProfile + otherMovieGenreProfile
#                
#                
#                similarity = self.computeSimilarity(thisMovieProfile, otherMovieProfile)
#                self.similarities[thisRating, otherRating] = similarity
#                self.similarities[otherRating, thisRating] = self.similarities[thisRating, otherRating]
                
        print("...done.")
                
        return self
    
    def computeSimilarity(self, movie1, movie2):
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
     
    
    def computeYearSimilarity(self, movie1, movie2):
         year1 = self.years[movie1]
         year2 = self.years[movie2]
         
         if (year1 == 0  or year2 == 0 ):
             return 1
             
         diff = abs(year1 - year2)
         sim = math.exp(-diff / 10.0)
         return sim
    
    
#    def estimate(self, u, i):
#        print("For user %d and item %d" % (int(self.trainset.to_raw_uid(u)), int(self.trainset.to_raw_iid(i))))
#        
##        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
##            raise PredictionImpossible('User and/or item is unknown.')
#        
#        thisMovieID = int(self.trainset.to_raw_iid(i))
##        
##        if (thisMovieID == 0):
##            raise PredictionImpossible('Item has no content data.')
#        
#        thisGenreProfile = self.createMovieGenreProfile(thisMovieID)
#        thisActorProfile = self.createMovieActorProfile(thisMovieID)
#        thisMovieProfile = thisActorProfile + thisGenreProfile
#        
#
#        neighbors = []
#        for rating in self.trainset.ur[u]:
#            otherMovieID = int(self.trainset.to_raw_iid(rating[0]))
#            
#            if(otherMovieID == 0):
#                continue
#                
#            otherGenreProfile = self.createMovieGenreProfile(otherMovieID)
#            otherActorProfile = self.createMovieActorProfile(otherMovieID)
#            otherMovieProfile = otherActorProfile + otherGenreProfile
#            
#            movieSimilarity = self.computeSimilarity(thisMovieProfile, otherMovieProfile)
#            textSimilarity  = self.computeSimilarity(self.scores.loc[thisMovieID], self.scores.loc[otherMovieID])
#                        
#            similarity = movieSimilarity * textSimilarity            
#            
#            if(movieSimilarity == 0):
#                similarity = textSimilarity
#            
#            if (textSimilarity == 0):
#                similarity = movieSimilarity
#            
#            neighbors.append((similarity, rating[1]))
#        
#        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
#        
#        simTotal = weightedSum = 0
#        for (simScore, rating) in k_neighbors:
#            if (simScore > 0):
#                simTotal += simScore
#                weightedSum += simScore * rating
#            
#        if (simTotal == 0):
#            raise PredictionImpossible('No neighbors')
#
#        predictedRating = weightedSum / simTotal
#        
#        return predictedRating
         
     
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
    
    
        
    