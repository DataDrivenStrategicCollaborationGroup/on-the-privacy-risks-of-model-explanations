#!/bin/sh
cd ..
model="attack_base"
epoch="15"

for dataset in "purchase" "texas" "cifar" "hospital" "adult"
do
experiment_name="${dataset}"
python train_attack.py ${experiment_name} ${model} "attack_output/${experiment_name}/gradient" --epochs ${epoch} --lr=1e-3 --lr_decay 1e-6   --use_explanation "gradient"  --verbose 0
done