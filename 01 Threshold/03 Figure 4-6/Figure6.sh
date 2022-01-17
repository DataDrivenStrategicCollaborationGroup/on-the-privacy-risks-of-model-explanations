#!/bin/sh
cd ..
cd ..
experiment_name="synthetic_big"
out_dir="../synthetic_output"
iteration_from=3
iteration_to=10
number_layers=3
number_nodes=100
verbose=0

for number_classes in 2 100
do
    for n_informative_features in  1 2 5 10 14 20 50 100 127 200 500 600 1000 2000 3072 5000 6000 10000
    do
        python synthetic_experiments.py ${experiment_name} ${out_dir}\
        --iteration_from ${iteration_from}\
        --iteration_to ${iteration_to}\
        --n_informative_features ${n_informative_features}\
        --number_classes ${number_classes}\
        --number_layers ${number_layers}\
        --number_nodes ${number_nodes}\
        --verbose ${verbose}
    done
done

cd ../SyntheticDatasets
experiment_name="synthetic_small"
out_dir="../synthetic_output"
iteration_from=0
iteration_to=10
number_layers=1
number_nodes=5
verbose=0

for number_classes in 2 100
do
    for n_informative_features in  1 2 5 10 14 20 50 100 127 200 500 600 1000 2000 3072 5000 6000 10000
    do
        python synthetic_experiments.py ${experiment_name} ${out_dir}\
        --iteration_from ${iteration_from}\
        --iteration_to ${iteration_to}\
        --n_informative_features ${n_informative_features}\
        --number_classes ${number_classes}\
        --number_layers ${number_layers}\
        --number_nodes ${number_nodes}\
        --verbose ${verbose}
    done
done
