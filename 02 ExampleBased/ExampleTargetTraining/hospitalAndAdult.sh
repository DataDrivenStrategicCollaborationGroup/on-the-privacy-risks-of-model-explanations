#!/bin/sh
cd ../..
dataset_size="2000"
model_type="logistic_regression"
for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
    for model_type in  "logistic_regression"
    do
        for dataset in "hospital" "adult"
        do
            echo ${seed}
            experiment_name="${dataset}_${model_type}"
            python train_record_target.py ${dataset} ${dataset_size} ${model_type}\
                   "record_output/${experiment_name}/${seed}" --seed ${seed}
        done
    done
done