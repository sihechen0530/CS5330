#!/bin/bash

python3 generate_params.py > arguments.txt

cat arguments.txt | xargs -I{} -P20 bash -c 'python3 experiment.py {}'

