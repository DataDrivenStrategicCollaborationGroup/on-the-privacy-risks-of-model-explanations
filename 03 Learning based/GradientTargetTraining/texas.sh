#!/bin/sh
cd ..
cd ..
experiment_name="texas"
dataset="texas"
model="texas_base"
    # batch_size = 512
    # epochs = 50
    # learning_rate = 0.01
    # lr_decay = 1e-7
for epoch in 50
do
    echo ${epoch}
    for seed in 0  1 2 3 4 5 6 7 8 9
    do
        echo ${seed}
        python train_target.py ${dataset} ${model} "target_output/${experiment_name}/${epoch}_${seed}" --epochs ${epoch} --seed ${seed} --lr 0.01 --lr_decay 1e-7
    done
done