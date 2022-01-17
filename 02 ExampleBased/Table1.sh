#!/bin/sh
cd ..
data_size=2000
for dataset in 'hospital' 'adult'
do
    python algorithm1.py ${dataset} ${data_size}
done
dataset='fishdog'
data_size=1800
python algorithm1.py ${dataset} ${data_size}  --seed_to 1
