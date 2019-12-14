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
    RandomForestClassifier(n_estimators=100),
    SVC(gamma='scale'),
    LogisticRegression(solver='lbfgs', multi_class='auto'),
    KNeighborsClassifier(),
    AdaBoostClassifier(),
    ExtraTreesClassifier(n_estimators=100),
    GradientBoostingClassifier(),
    DecisionTreeClassifier(),
    DummyClassifier('most_frequent'),
]

regressors = [
    RandomForestRegressor(n_estimators=100),
    SVR(gamma='scale'),
    LinearRegression(),
    KNeighborsRegressor(),
    AdaBoostRegressor(),
    ExtraTreesRegressor(n_estimators=100),
    GradientBoostingRegressor(),
    DecisionTreeRegressor(),
    DummyRegressor('mean'),
]

not_multi = [
    AdaBoostRegressor,
    GradientBoostingRegressor,
    SVR,
]

if lightgbm:
    classifiers.insert(0, lightgbm.LGBMClassifier())
    regressors.insert(0, lightgbm.LGBMRegressor())
    not_multi.append(lightgbm.LGBMRegressor)

if xgboost:
    classifiers.insert(0, xgboost.XGBClassifier())
    regressors.insert(0, xgboost.XGBRegressor(objective='reg:squarederror'))
    not_multi.append(xgboost.XGBRegressor)

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

    def __init__(self, names=None, task=None, multi=False):
        if task == 'classification':
            items = classifiers
        elif task == 'regression':
            items = regressors
        else:
            items = {}
        if names is not None:
            items = {k: items[k] for k in names}
        self.multi = multi
        super(Estimators, self).__init__(items)

    def get(self, name):
        estimator = clone(self.items[name])
        if self.multi:
            estimator = make_multi(estimator)
        return estimator
