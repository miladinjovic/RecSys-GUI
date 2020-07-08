import pandas as pd
import os
import csv
import sys

from surprise import Dataset
from surprise import Reader
from surprise.model_selection import PredefinedKFold

from collections import defaultdict

class YahooDataset:

    
    ratingsTrainPath = 'R4/ydata-ymovies-user-movie-ratings-train-v1_0.txt'
    ratingsTestPath = 'R4/ydata-ymovies-user-movie-ratings-test-v1_0.txt'
    usersPath = 'R4/ydata-ymovies-user-demographics-v1_0.txt'
    moviesPath = 'R4/movie_db_yoda'
    fullSetPath = 'R4/fullSet.txt'
    
    def loadFullSet(self):
        reader = Reader(line_format='user item rating timestamp' ,sep='\t', skip_lines=0)
        fullDataset = Dataset.load_from_file(self.fullSetPath, reader=reader)

        return fullDataset.build_full_trainset()
    
    def loadYahooPandasFullDataFrame(self):
        os.chdir(os.path.dirname(sys.argv[0]))
        data = pd.read_csv(self.fullSetPath, delimiter = '\t', header=None, 
                          names = ["userId", "movieId","rating"], usecols =["userId", "movieId", "rating"])
        return data

    
    def loadDemographicsData(self):
        os.chdir(os.path.dirname(sys.argv[0]))
        data = pd.read_csv(self.usersPath, delimiter='\t', header=None,
                               names=["userId", "year", "gender"])
        return data
    
    def loadNormalizedData(self, data=None):
        if data is None:
            data = self.loadYahooPandasTrainingDataframe()
        
        normalizedByUser, usersAverage = self.normalizeByUser(data)
        
        allNormalized, itemsAverage = self.normalizeByItem(normalizedByUser)
        
        return (allNormalized, usersAverage, itemsAverage)
        
    
    def normalizeByUser(self, data=None ):
        if data is None:
            data = self.loadYahooPandasTrainingDataframe()
        
        #data.rating = data.iloc[:, [0, 2]].set_index("userId").transform(lambda p: p-usersAverage.loc[p.name, "rating"] ,axis=1).reset_index().rating
#        usersAverage =  data.iloc[:, [0, 2]].groupby("userId").mean()
        normalizedByUser = data.set_index("movieId").groupby("userId").transform(lambda p: p-p.mean()).reset_index()
        normalizedByUser.insert(0, "userId", data.userId)
        
        return normalizedByUser

    
    def normalizeByItem(self, data=None):
        if data is None:
            data = self.loadYahooPandasTrainingDataframe()
        
#        itemsAverage = data.iloc[:, [1,2]].groupby("movieId").mean()
        normalizedByItem = data.set_index("userId").groupby("movieId").transform(lambda p: p-p.mean()).reset_index()
        normalizedByItem.insert(1, "movieId", data.movieId)
        
        return normalizedByItem
    
    def loadFromPandas(self, frame):
         reader = Reader(rating_scale=(frame["rating"].min(), frame["rating"].max()))
         data = Dataset.load_from_df(frame, reader=reader)
         return data
        
        
    def loadYahooPandasTrainingDataFrame(self):
        os.chdir(os.path.dirname(sys.argv[0]))
        data = pd.read_csv(self.ratingsTrainPath, delimiter = '\t', header=None, 
                          names = ["userId", "movieId","rating"], usecols =["userId", "movieId", "rating"])
        return data
    
    def loadYahooPandasTestDataFrame(self):
        os.chdir(os.path.dirname(sys.argv[0]))
        data = pd.read_csv(self.ratingsTestPath, delimiter = '\t', header=None, 
                          names = ["userId", "movieId","rating"], usecols =["userId", "movieId", "rating"])
        return data
    
    def getTestDataGlobalMean(self):
        testData = self.loadYahooPandasTestDataFrame()
        return testData["rating"].mean()
    
    def loadYahooDataset(self):
        
        os.chdir(os.path.dirname(sys.argv[0]))
        reader = Reader(line_format='user item rating timestamp' ,sep='\t', skip_lines=0)
        
        data = Dataset.load_from_folds([(self.ratingsTrainPath, self.ratingsTestPath)], reader=reader)
        pkf = PredefinedKFold()
        
        return pkf.split(data)
    
    def loadYahooTrainDataset(self):

        os.chdir(os.path.dirname(sys.argv[0]))
        reader = Reader(line_format='user item rating timestamp' ,sep='\t', skip_lines=0)
        ratingsTrainDataset = Dataset.load_from_file(self.ratingsTrainPath, reader=reader)

        return ratingsTrainDataset.build_full_trainset()
    
    def loadYahooTestDataset(self):
        
         os.chdir(os.path.dirname(sys.argv[0]))
         reader = Reader(line_format='user item rating timestamp' ,sep='\t', skip_lines=0)
         ratingsTestDataset = Dataset.load_from_file(self.ratingsTestPath, reader=reader)
         
         return ratingsTestDataset.construct_testset(ratingsTestDataset.raw_ratings)
    
    def loadMovies(self):
        # movieID_to_info = defaultdict(str)
        movieID_to_info = defaultdict(dict)
        with open(self.moviesPath, newline='') as csvfile:
             movieReader = csv.reader(csvfile, delimiter='\t')
             for row in movieReader:
                 movieID = row[0]
                 title = row[1]
                 genreList = row[10]
                 if genreList != "\\N":
                     genreList = genreList.split("|")
                     genreList = ", ".join(genreList)

                 actorList = row[16]
                 if actorList != "\\N":
                     actorList = actorList.split("|")
                     actorList = ", ".join(actorList)

                 synopsis = row[2]

                 movieID_to_info[movieID] = {"Title": title,
                                             "Genres": genreList,
                                             "Actors": actorList,
                                             "Synopsis": synopsis}
                 
        return movieID_to_info

    def getHistory(self, user, k=10):
        ratings = self.loadYahooPandasFullDataFrame()
        ratingsByUser = ratings[ratings["userId"] == user].sort_values(by=['rating'], ascending=False)

        return zip(ratingsByUser["movieId"][:10].astype("str"), ratingsByUser["rating"][:10])
        # return ratingsByUser[["movieId", "rating"]].head(10)
                 