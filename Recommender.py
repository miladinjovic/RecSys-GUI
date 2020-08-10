# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:45:15 2020

@author: Ja
"""

import itertools

from surprise import KNNBaseline
from collections import defaultdict
import math
from YahooDataset import YahooDataset
from surprise import SVD, KNNBaseline
from WeightedHybridAlgorithm import WeightedHybridAlgorithm
from knnRecAlgorithm import knnRecAlgorithm
from BPRecommender import BPRecommender
from ModelManager import ModelManager

from tmdbv3api import TMDb
from tmdbv3api import Movie
from tmdbv3api import TV

import numpy as np
from faker import Faker

class Recommender:

    def __init__(self):
        self.yd = YahooDataset()
        self.movieID_to_info = self.yd.loadMovies()
        self.fullTrainSet  = self.yd.loadFullSet()

        self.models = ModelManager().getAllModels()
        self.fullTrainSet.rating_scale = (1, 13)

    
    def _buildAntiTestSetForUser(self, testSubject):
        trainset = self.fullTrainSet
        fill = trainset.global_mean

        anti_testset = []
        
        user_items = set([j for (j, _) in trainset.ur[testSubject]])
        anti_testset += [(trainset.to_raw_uid(testSubject), trainset.to_raw_iid(i), fill) for
                             i in trainset.all_items() if
                             i not in user_items]
        return anti_testset
    
    def _getTopNForUser(self, predictions, n=10, minimumRating=8.0):
        
        topN = []
        
        for userID, movieID, actualRating, estimatedRating, _ in predictions:
            if(estimatedRating >= minimumRating):
                topN.append( (movieID, estimatedRating) )
        
        topN.sort(key=lambda x: x[1], reverse = True)
        
        return  topN[:n]

    def deleteModel(self, algo, model):
        path = "models/" + algo + "/" + model
        try:
            ModelManager().deleteModel(path)
        except OSError as e:
            return "error"

        self.models[algo].remove(model)

        return "success"

    def getHistory(self, user, k=10):
        history = [(self.getAdditionalData(movieId), rating) for movieId, rating in self.yd.getHistory(user) if movieId != "0"]
        return history

    def getAdditionalData(self, movieId):
        if "Url" not in self.movieID_to_info[movieId]:
            tmdb = TMDb()
            tmdb.api_key = '66a11b9611008995220d8300279b6011'

            tmdb.language = 'en'
            tmdb.debug = True
            base = "https://image.tmdb.org/t/p/w200"
            url = ""
            title = self.movieID_to_info[movieId]["Title"][:-6]
            search = Movie().search(title)

            if search != []:
                url = search[0].poster_path
            else:
                search = TV().search(title)
                if search != []:
                    url = search[0].poster_path

            self.movieID_to_info[movieId]["Url"] = base + url


        return self.getChartData(movieId)

    def getChartData(self, movieId):
        if "Stat" not in self.movieID_to_info[movieId]:
            trainSet = self.fullTrainSet

            innerId = trainSet.to_inner_iid(movieId)
            ratings = defaultdict(int)
            for _, rating in trainSet.ir[innerId]:
                ratings[int(rating)] += 1

            ratings = dict(sorted(ratings.items()))

            self.movieID_to_info[movieId]["Stat"] = (list(ratings.values()), list(ratings.keys()))

        return movieId

    def recommend(self, params):
        user = self.fullTrainSet.to_inner_uid(params["user"])

        antiTestSet = self._buildAntiTestSetForUser(user)

        algo = params["algorithm"]
        path = "models/" + algo

        if algo == "svd":
            if "models" not in params.keys():
                args = {
                    "random_state": 0,
                    "reg_all": float(params["rr"]),
                    "lr_all": float(params["lr"]),
                    "n_epochs": int(params["ne"]),
                    "n_factors": int(params["factors"])
                }

                svd = SVD(**args)
                svd = svd.fit(self.fullTrainSet)
                predictions = svd.test(antiTestSet)

                if "name" in params.keys():
                    mm = ModelManager()
                    name = params["name"]
                    path = path + "/" + name
                    mm.saveModel(svd, path)
                    self.models[algo].append(name)
            else:
                mm = ModelManager()
                model = params["models"]
                path = path + "/" + model

                svd, _ = mm.loadModel(path)
                predictions = svd.test(antiTestSet)

            topN = self._getTopNForUser(predictions)
            topN = [(self.getAdditionalData(movieId), round(estimated, 2)) for movieId, estimated in topN]

        elif algo == "knnItemBaseline":

            if "models" not in params.keys():
                args = {
                    "sim_options" : {'name': 'cosine', 'user_based': False},
                    "k": int(params["k"])
                }

                knn = KNNBaseline(**args)
                knn = knn.fit(self.fullTrainSet)
                predictions = knn.test(antiTestSet)

                if "name" in params.keys():
                    mm = ModelManager()
                    name = params["name"]
                    path = path + "/" + name
                    mm.saveModel(knn, path)
                    self.models[algo].append(name)
            else:
                mm = ModelManager()
                model = params["models"]
                path = path + "/" + model

                knn, _ = mm.loadModel(path)
                predictions = knn.test(antiTestSet)

            topN = self._getTopNForUser(predictions)
            topN = [(self.getAdditionalData(movieId), round(estimated, 2)) for movieId, estimated in topN]

        elif algo == "weightedHybrid":

            svd = SVD(random_state=0, reg_all=0.1, lr_all=0.003, n_factors=30, verbose=False)
            knn = KNNBaseline(sim_options={'name': 'cosine', 'user_based': False}, k=150)
            weightedHybrid = WeightedHybridAlgorithm(svd, knn, weights=[0.6, 0.4])
            weightedHybrid.fit(self.fullTrainSet)
            predictions = weightedHybrid.test(antiTestSet)
            topN = self._getTopNForUser(predictions)
            topN = [(self.getAdditionalData(movieId), round(estimated, 2)) for movieId, estimated in topN]

        elif algo == "userCollaborative":

            if "models" not in params.keys():
                args = {
                    "k": int(params["k"]),
                    "sim_options": {'name': 'cosine', 'user_based': True}
                }

                knn = knnRecAlgorithm(**args)
                knn = knn.fit(self.fullTrainSet)
                predictions = knn.test(antiTestSet)

                if "name" in params.keys():
                    mm = ModelManager()
                    name = params["name"]
                    path = path + "/" + name
                    mm.saveModel(knn, path)
                    self.models[algo].append(name)
            else:
                mm = ModelManager()
                model = params["models"]
                path = path + "/" + model

                knn, _ = mm.loadModel(path)
                predictions = knn.test(antiTestSet)

            topN = self._getTopNForUser(predictions, minimumRating=0.0)
            topN = [(self.getAdditionalData(movieId), round(estimated, 2)) for movieId, estimated in topN]


        elif algo == "bpr":

            if "models" not in params.keys():

                args = {
                    "reg": float(params["rr"]),
                    'learning_rate': float(params["lr"]),
                    'n_iters': int(params["ni"]),
                    'n_factors': int(params["factors"]),
                    'batch_size': 100
                }

                bpr = BPRecommender(args)
                bpr = bpr.fit()

                if "name" in params.keys():
                    mm = ModelManager()
                    name = params["name"]
                    path = path + "/" + name
                    mm.saveBprModel(bpr, path)
                    self.models[algo].append(name)
            else:
                mm = ModelManager()
                model = params["models"]
                path = path + "/" + model
                bpr = mm.loadBprModel(path)

            topN = bpr.recommend(user)

            topN = [(self.getAdditionalData(movieId), None) for movieId in topN]

        return topN

    def getRandomUsers(self):
        randomIds = np.random.randint(1, self.fullTrainSet.n_users, 30)

        faker = Faker()
        randomUsers = {}
        for id in randomIds:
            randomUsers[str(id)] = faker.name()

        return randomUsers



