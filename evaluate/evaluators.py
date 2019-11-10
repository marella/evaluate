import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .estimators import Estimators
from .preprocessors import Preprocessors
from .utils import find_columns


class Evaluator():

    def __init__(self,
                 task,
                 data,
                 test_data=None,
                 split=.2,
                 columns=None,
                 estimators=None,
                 preprocessors=None):
        if test_data is None:
            x, y = data
            stratify = y if task == 'classification' else None
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, stratify=stratify, test_size=split)
            data = x_train, y_train
            test_data = x_test, y_test
        x_train, y_train = data
        multi = len(y_train.shape) > 1
        if columns is None:
            columns = find_columns(x_train)
        if estimators is None:
            estimators = Estimators(task=task)
        if preprocessors is None:
            preprocessors = Preprocessors()
        estimators.multi = multi
        preprocessors.columns = columns
        self.task = task
        self.data = data
        self.test_data = test_data
        self.columns = columns
        self.multi = multi
        self.estimators = estimators
        self.preprocessors = preprocessors

    def evaluate(self, seed=None):
        if seed is None:
            seed = int(time.time()) % 1000
        names_e = self.estimators.names
        names_p = self.preprocessors.names
        scores = defaultdict(list)
        for name_e in names_e:
            estimator = self.estimators.get(name_e)
            for name_p in names_p:
                model = self.preprocessors.apply(name_p, estimator)
                # It is important to use same seed for an estimator to be able
                # to compare results across different preprocessing pipelines
                np.random.seed(seed)
                model.fit(*self.data)
                score = model.score(*self.test_data)
                scores[name_p].append(score)
        return pd.DataFrame(scores, index=names_e)


def evaluate(**kwargs):
    return Evaluator(**kwargs).evaluate()
