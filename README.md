A tool to evaluate the performance of various machine learning algorithms and preprocessing steps to find a good baseline for a given task.

## Installation

```sh
pip install evaluate
```

## Usage

```py
import evaluate

results = evaluate(task='classification',
                  data=(x_train, y_train),
                  test_data=(x_test, y_test))

results['test_score'].plot.bar()
```
