import argparse
from datasets import create_attack_dataset
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Run correlation analysis')
parser.add_argument('experiment', type=str)
parser.add_argument('out_dir', type=str, help='Folder to save results)')
parser.add_argument('--use_loss', default=False, action='store_true',
                    help='Calculate the correlation between loss and membership.')
args = parser.parse_args()


def correlation_membership_signals():
    print(args.experiment)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    result_file_name = os.path.join(args.out_dir, 'result.csv')
    result_file = open(result_file_name, 'w')
    result_file.write("target_dir,target_train_acc,target_test_acc,corr_var_norm,corr_membership_var,corr_mem_norm\n")

    used_features = ["gradient", "prediction"]
    one_norm_features = ["gradient"]
    variance_features = ["prediction"]

    for directory in os.listdir("target_output/{}/".format(args.experiment)):
        print(directory)
        path_to_dataset = "target_output/{}/{}".format(args.experiment, directory)
        print("Creating dataset")
        train_dataset, test_dataset = create_attack_dataset(path_to_dataset, used_features, 0,
                                                            one_norm_features=one_norm_features,
                                                            variance_features=variance_features)
        one_norm = np.concatenate([train_dataset.x[0], test_dataset.x[0]])
        variance = np.concatenate([train_dataset.x[1], test_dataset.x[1]])
        membership = np.concatenate([train_dataset.y, test_dataset.y])
        correlation_1 = np.corrcoef(one_norm, variance)[0, 1]
        correlation_2 = np.corrcoef(membership, variance)[0, 1]
        correlation_3 = np.corrcoef(membership, one_norm)[0, 1]
        target_log = pd.read_csv(r'' + path_to_dataset + '/log.csv', header=0, delimiter=";")
        target_train_acc = np.asarray(target_log["acc"])[-1]
        target_test_acc = np.asarray(target_log["val_acc"])[-1]
        result_file.write("{},{},{},{},{},{}\n".format(
            directory, target_train_acc, target_test_acc, correlation_1, correlation_2, correlation_3)
        )
        print("{},{},{},{},{},{}\n".format(
            directory, target_train_acc, target_test_acc,  correlation_1, correlation_2, correlation_3))
    result_file.close()


def correlation_membership_loss():
    print(args.experiment)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    result_file_name = os.path.join(args.out_dir, 'corr_mem_loss.csv')
    result_file = open(result_file_name, 'w')
    result_file.write("target_dir,target_train_acc,target_test_acc,corr_mem_loss\n")

    used_features = ["prediction", "label"]
    one_norm_features = []
    variance_features = []

    for directory in os.listdir("target_output/{}/".format(args.experiment)):
        print(directory)
        path_to_dataset = "target_output/{}/{}".format(args.experiment, directory)
        print("Creating dataset")
        train_dataset, test_dataset = create_attack_dataset(path_to_dataset, used_features, 0,
                                                            one_norm_features=one_norm_features,
                                                            variance_features=variance_features)

        prediction = np.concatenate([train_dataset.x[0], test_dataset.x[0]])
        label = np.concatenate([train_dataset.x[1], test_dataset.x[1]])
        print(label.shape, prediction.shape)
        loss = np.abs(prediction-label).sum(axis=1)
        membership = np.concatenate([train_dataset.y, test_dataset.y])
        correlation = np.corrcoef(loss, membership)[0, 1]
        target_log = pd.read_csv(r'' + path_to_dataset + '/log.csv', header=0, delimiter=";")
        target_train_acc = np.asarray(target_log["acc"])[-1]
        target_test_acc = np.asarray(target_log["val_acc"])[-1]
        result_file.write("{},{},{},{}\n".format(
            directory, target_train_acc, target_test_acc, correlation)
        )
        print("{},{},{},{}\n".format(
            directory, target_train_acc, target_test_acc,  correlation))
    result_file.close()


def main():
    if args.use_loss:
        correlation_membership_loss()
    else:
        correlation_membership_signals()


if __name__ == "__main__":
    main()
