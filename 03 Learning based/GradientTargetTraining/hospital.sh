#!/bin/sh
cd ..
cd ..
experiment_name="hospital"
dataset="hospital"
model="hospital_base"

for epoch in  1000
do
    echo ${epoch}
    for seed in 0 1 2 3 4 5 6 7 8 9 10
    do
        echo ${seed}
        python train_target.py ${dataset} ${model} "target_output/${experiment_name}/${epoch}_${seed}" --epochs ${epoch} --seed ${seed} --lr 0.001 --lr_decay 1e-6
    done
done
