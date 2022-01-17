#!/bin/sh
cd ..
cd ..
experiment_name="adult"
dataset="adult"
model="adult_base"
for epoch in  10
do
    echo ${epoch}
    for seed in 0 1 2 3 4 5 6 7 8 9 10
    do
        echo ${seed}
        python train_target.py ${dataset} ${model} "target_output/${experiment_name}/${epoch}_${seed}" --epochs ${epoch} --seed ${seed} --lr 0.01 --lr_decay 1e-7 --batch 10000
    done
done