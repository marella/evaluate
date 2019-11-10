import inspect

import pandas as pd


def find_columns(x, categorical_max=10, drop=None):
    if not isinstance(x, pd.DataFrame):
        return {'numeric': list(range(x.shape[1]))}
    numeric = x.select_dtypes(include='number').columns
    categorical = x.select_dtypes(exclude='number')
    categorical = categorical.columns[categorical.nunique() <= categorical_max]
    if drop:
        numeric = numeric.drop(drop, errors='ignore')
        categorical = categorical.drop(drop, errors='ignore')
    columns = dict(numeric=list(numeric), categorical=list(categorical))
    return {k: v for k, v in columns.items() if v}


def callable_module(fn):
    caller = inspect.stack()[1][0]
    module = inspect.getmodule(caller)

    class Module(module.__class__):

        def __call__(self, *args, **kwargs):
            return fn(*args, **kwargs)

    module.__class__ = Module


class Dict():

    def __init__(self, items=None):
        items = {} if items is None else items
        self.items = {**items}

    @property
    def names(self):
        return list(self.items.keys())
