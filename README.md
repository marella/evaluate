A tool to evaluate the performance of various machine learning algorithms and preprocessing steps on a given task to find a good baseline.

## Installation

```sh
pip install evaluate
```

## Usage

```py
import evaluate

scores = evaluate(task='classification',
                  data=(x_train, y_train),
                  test_data=(x_test, y_test))

scores.plot.bar()
```
