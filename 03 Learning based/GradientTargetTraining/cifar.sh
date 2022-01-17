#!/bin/sh
cd ..
cd ..
experiment_name="cifar"
dataset="cifar_10"
model="cifar_base"

for epoch in  50
do
    echo ${epoch}
    for seed in 0 1 2 3 4 5 6 7 8 9
    do
        echo ${seed}
        python train_target.py ${dataset} ${model} "target_output/${experiment_name}/${epoch}_${seed}" --epochs ${epoch} --seed ${seed} --lr 0.001 --batch 512
    done
done