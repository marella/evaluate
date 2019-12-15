from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import clone

from .utils import Dict

base = {
    'numeric': [
        SimpleImputer(strategy='median', add_indicator=True),
    ],
    'categorical': [
        SimpleImputer(strategy='constant'),
        OneHotEncoder(handle_unknown='ignore'),
    ],
    'ordinal': [
        SimpleImputer(strategy='constant'),
        OrdinalEncoder(),
    ],
}

preprocessors = {
    'n': {
        'numeric': [],
    },
    'n:s': {
        'numeric': [StandardScaler()],
    },
    'c': {
        'categorical': [],
    },
    'o': {
        'ordinal': [],
    },
    't:c': {
        'text': [CountVectorizer(max_features=20000)],
    },
    't:t': {
        'text': [TfidfVectorizer(max_features=20000)],
    },
    't:c=2': {
        'text': [CountVectorizer(ngram_range=(1, 2), max_features=20000)],
    },
    't:t=2': {
        'text': [TfidfVectorizer(ngram_range=(1, 2), max_features=20000)],
    },
}

for _, preprocessor in preprocessors.items():
    overwrite = preprocessor.pop('overwrite', False)
    remainder = preprocessor.pop('remainder', 'drop')
    for t in preprocessor:
        steps = [] if overwrite else base.get(t, [])
        steps = steps + preprocessor[t]
        preprocessor[t] = make_pipeline(*steps)
    preprocessor['remainder'] = remainder


class Preprocessors(Dict):

    def __init__(self, names=None, columns=None):
        if names is None:
            names = ['n,c,o', 'n:s,c,o']
        if columns is None:
            columns = {}
        self.columns = columns
        items = {k: self.make(k) for k in names}
        super(Preprocessors, self).__init__(items)

    def get(self, name):
        preprocessor = self.items[name]
        remainder = preprocessor.get('remainder', 'drop')
        transformers = [(t, clone(preprocessor[t]), self.columns[t])
                        for t in self.columns if t in preprocessor]
        return ColumnTransformer(transformers, remainder=remainder)

    def apply(self, name, estimator):
        preprocessor = self.get(name)
        estimator = clone(estimator)
        return Pipeline([('preprocessor', preprocessor),
                         ('estimator', estimator)])

    def make(self, name):
        preprocessor = {}
        for n in name.split(','):
            preprocessor.update(preprocessors[n.strip()])
        return preprocessor
