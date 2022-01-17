#!/bin/sh
data_folder=data
mkdir ${data_folder}

mkdir ${data_folder}/adult
curl https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data --output  ${data_folder}/adult/adult.data
curl https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names --output  ${data_folder}/adult/adult.names
curl https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test --output  ${data_folder}/adult/adult.test
# For some reason the lines in adult.test all end with a period, so we remove that
sed -i 's/K./K/' ${data_folder}/adult/adult.test

mkdir ${data_folder}/purchase
curl https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz --output  ${data_folder}/purchase/purchase.tgz
tar zxvf ${data_folder}/purchase/purchase.tgz -C ${data_folder}/purchase
mv ${data_folder}/purchase/dataset_purchase ${data_folder}/purchase/purchase.csv
rm ${data_folder}/purchase/purchase.tgz

mkdir ${data_folder}/texas
curl https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz --output  ${data_folder}/texas/texas.tgz
tar zxvf ${data_folder}/texas/texas.tgz -C ${data_folder}/
rm ${data_folder}/texas/texas.tgz

mkdir ${data_folder}/hospital
curl https://worksheets.codalab.org/rest/bundles/0xfabc11ae87864c65a65d548562b8e409/contents/blob/\
     --output  ${data_folder}/hospital/hospital

mkdir ${data_folder}/fishdog
curl https://worksheets.codalab.org/rest/bundles/0x550cd344825049bdbb865b887381823c/contents/blob/dataset_dog-fish_train-900_test-300.npz\
     --output  ${data_folder}/fishdog/dataset_dog-fish_train-900_test-300.npz