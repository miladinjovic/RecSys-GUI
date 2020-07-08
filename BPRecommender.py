import warnings

warnings.simplefilter("ignore")

from YahooDataset import YahooDataset
from BPRAlgorithm import BPRAlgorithm
from scipy.sparse import csr_matrix
import numpy as np

class BPRecommender:

    def __init__(self, params):
        self.params = params

    def _create_matrix(self, data):
        data['bprRating'] = np.where(data['rating'] >= 0, 1, 0)

        for col in ('movieId', 'userId', 'bprRating'):
            data[col] = data[col].astype('category')

        data["codes"] = data['movieId'].cat.codes
        ratings = csr_matrix((data['bprRating'], (data['userId'].cat.codes, data['movieId'].cat.codes)))
        ratings.eliminate_zeros()
        return ratings, data

    def _prepareData(self):
        yd = YahooDataset()
        df = yd.loadYahooPandasFullDataFrame()
        ratingCol = df["rating"]
        df = yd.normalizeByUser(df)
        X, df = self._create_matrix(df)
        df["rating"] = ratingCol

        return X, df

    def fit(self):
        self.X, self.df = self._prepareData()
        # bpr_params = {'reg': 0.02,
        #               'learning_rate': 0.2,
        #               'n_iters': 600,
        #               'n_factors': 15,
        #               'batch_size': 100,
        #               'df': self.df}
        self.params['df'] = self.df

        self.bpr = BPRAlgorithm(**self.params)
        self.bpr = self.bpr.fit(self.X)

        return self

    def recommend(self, user):
        top_n = self.bpr._recommend_user(self.X, user, 10)
        top_n = [ self.df["movieId"].cat.categories[rec].astype("str") for rec in top_n ]

        return top_n
