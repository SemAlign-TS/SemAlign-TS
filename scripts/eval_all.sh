#!/usr/bin/env bash
set -e

for d in etth1 ettm1 electricity exchange traffic weather
do
  echo "Evaluating diffusion baseline on ${d}"
  python evaluate.py --dataset $d --mode diff_train

  echo "Evaluating OSRA model on ${d}"
  python evaluate.py --dataset $d --mode osra
done