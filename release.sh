#!/usr/bin/env bash

clean='rm -r dist *.egg-info'

$clean
python3 setup.py sdist
twine upload dist/*
$clean
