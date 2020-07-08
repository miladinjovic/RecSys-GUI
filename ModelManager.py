import os

import json
import pickle
from surprise import dump
import threading


class ModelManager():

    def saveModel(self, model, path):
      thread = threading.Thread(target=self._save, args=(model, path))
      thread.start()

    def _save(self, model, path):
        dump.dump(path, model)

    def loadModel(self, path):
        return dump.load(path)

    def getAllModels(self):
        models = {}
        algos = ["svd", "knnItemBaseline", "userCollaborative", "bpr"]

        for algo in algos:
            models[algo] = os.listdir("models/" + algo)

        return models

    def saveBprModel(self, model, path):
        thread = threading.Thread(target=self._saveBpr, args=(model, path))
        thread.start()

    def _saveBpr(self, model, path):
        with open(path, "wb") as file:
            pickle.dump(model, file)


    def loadBprModel(self, path):
        with open(path, "rb") as file:
            conf = pickle.load(file)

        return conf

    def deleteModel(self, path):
        try:
            os.remove(path)
        except OSError as e:
            raise(e)






