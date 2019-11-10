from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import clone

from .utils import Dict

try:
    import xgboost
except:
    xgboost = None

try:
    import lightgbm
except:
    lightgbm = None

classifiers = [
    AdaBoostClassifier(),
    ExtraTreesClassifier(n_estimators=100),
    GradientBoostingClassifier(),
    LogisticRegression(solver='lbfgs', multi_class='auto'),
    GaussianNB(),
    KNeighborsClassifier(),
    SVC(gamma='scale'),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100),
    DummyClassifier('most_frequent'),
]

regressors = [
    AdaBoostRegressor(),
    ExtraTreesRegressor(n_estimators=100),
    GradientBoostingRegressor(),
    LinearRegression(),
    KNeighborsRegressor(),
    SVR(gamma='scale'),
    DecisionTreeRegressor(),
    RandomForestRegressor(n_estimators=100),
    DummyRegressor('mean'),
]

not_multi = [
    AdaBoostRegressor,
    GradientBoostingRegressor,
    SVR,
]

if xgboost:
    classifiers.append(xgboost.XGBClassifier())
    regressors.append(xgboost.XGBRegressor(objective='reg:squarederror'))
    not_multi.append(xgboost.XGBRegressor)

if lightgbm:
    classifiers.append(lightgbm.LGBMClassifier())
    regressors.append(lightgbm.LGBMRegressor())
    not_multi.append(lightgbm.LGBMRegressor)

not_multi = set(not_multi)


def supports_multi(estimator):
    return estimator.__class__ not in not_multi


def make_multi(estimator):
    if supports_multi(estimator):
        return estimator
    else:
        return MultiOutputRegressor(estimator)


def add_names(items):
    result = {}
    for item in items:
        if isinstance(item, tuple):
            name, item = item
        else:
            name = item.__class__.__name__
        result[name] = item
    return result


classifiers = add_names(classifiers)
regressors = add_names(regressors)


class Estimators(Dict):

    def __init__(self, estimators=None, task=None, multi=False):
        if estimators is None:
            if task == 'classification':
                estimators = classifiers
            elif task == 'regression':
                estimators = regressors
            else:
                estimators = {}
        self.task = task
        self.multi = multi
        super(Estimators, self).__init__(estimators)

    def get(self, name):
        estimator = clone(self.items[name])
        if self.multi:
            estimator = make_multi(estimator)
        return estimator
