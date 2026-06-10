#!/usr/bin/env bash
set -e

for d in etth1 ettm1 electricity exchange traffic weather
do
  echo "Training diffusion model on ${d}"
  python diff_train.py --dataset $d

  echo "Fine-tuning OSRA model on ${d}"
  python osra_train.py --dataset $d
done