#!/bin/bash

for file in ./*jupyter.py; do
    jupytext --from py:percent --to ipynb "$file"
done