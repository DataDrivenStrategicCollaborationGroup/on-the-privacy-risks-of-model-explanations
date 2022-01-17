# This file trains a target network and creates a dataset on which the attack can be trained.
# Written by Martin Strobel
# based on https://github.com/bearpaw/pytorch-classification
import argparse
from datasets import create_attack_dataset
from architectures import get_attack_model
from keras import optimizers
from keras.callbacks import CSVLogger
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle
from constants import EXPLANATIONS, ARCHITECTURES
# noinspection PyUnresolvedReferences
import setGPU

parser = argparse.ArgumentParser(description='Train attack model')
parser.add_argument('experiment', type=str)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('out_dir', type=str, help='folder to save log)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=512, type=int, metavar='N',
                    help='batchsize (default: 512)')
parser.add_argument('--use_label', default=False, action='store_true',
                    help='Use the true label to train attack')
parser.add_argument('--use_loss', default=False, action='store_true',
                    help='Use the loss to train attack')
parser.add_argument('--use_predicted_label', default=False, action='store_true',
                    help='Use the predicted label to train attack')
parser.add_argument('--use_prediction', default=False, action='store_true',
                    help='Use the full prediction label to train attack')
parser.add_argument('--use_explanation', type=str, choices=EXPLANATIONS, default="none",
                    help='Use an explanation to train attack')
parser.add_argument('--use_neural_network', default=True, action='store_false',
                    help='Use a neural network to attack')
parser.add_argument('--use_1norm', default=False, action='store_true',
                    help='Use the 1 norm of the gradient')
parser.add_argument('--normalize_gradient', default=False, action='store_true',
                    help='Normalize the gradient wrt 1norm')
parser.add_argument('--save_predictions', default=False, action='store_true',
                    help='Save the predictions of the attacker')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_decay', type=float, default=1e-7,
                    help='LR decay.')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=None, help='Random seed for train test split')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--verbose', default=2, type=int, help='Output of training', choices=[0, 1, 2])
args = parser.parse_args()


def main():
    print(args.experiment)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    result_file_name = os.path.join(args.out_dir, 'result.csv')
    result_file = open(result_file_name, 'w')
    result_file.write("target_dir,target_train_acc,target_test_acc,attack_train_acc,attack_test_acc\n")

    if not args.seed:
        args.seed = np.random.randint(low=0, high=1000000)

    used_features, one_norm_features = [], []
    if args.use_label:
        used_features.append("label")
    if args.use_loss:
        used_features.append("loss")
    if args.use_predicted_label:
        used_features.append("predicted_label")
    if args.use_prediction:
        used_features.append("prediction")
    if args.use_explanation != "none":
        used_features.append(args.use_explanation)
    if args.use_1norm:
        used_features = ["gradient"]
        one_norm_features = ["gradient"]

    for directory in os.listdir("target_output/{}/".format(args.experiment)):
        print(directory)
        path_to_dataset = "target_output/{}/{}".format(args.experiment, directory)
        print("Creating dataset")
        train_dataset, test_dataset = create_attack_dataset(path_to_dataset, used_features, int(args.seed),
                                                            one_norm_features=one_norm_features,
                                                            normalize_gradient=args.normalize_gradient)

        print("Training the attack model")

        if args.use_neural_network:

            shape_dict = {}
            for i, feature in enumerate(used_features):
                shape_dict[feature] = len(train_dataset.x[i][0])
            print(shape_dict)
            model = get_attack_model(args.arch, used_features, shape_dict)
            log_file_name = os.path.join(args.out_dir, '{}_log.csv'.format(directory))
            csv_logger = CSVLogger(log_file_name, append=True, separator=';')
            optimizer = optimizers.Adagrad(lr=args.lr, decay=args.lr_decay)
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            model.fit(train_dataset.x, train_dataset.y, epochs=args.epochs,
                      validation_data=(test_dataset.x, test_dataset.y), callbacks=[csv_logger], verbose=args.verbose)

            test_acc = model.evaluate(test_dataset.x, test_dataset.y, verbose=args.verbose)[-1]
            train_acc = model.evaluate(train_dataset.x, train_dataset.y, verbose=args.verbose)[-1]
        else:
            print("Use a decision tree")
            model = DecisionTreeClassifier(max_depth=1)
            train_dataset.x = np.concatenate(train_dataset.x, axis=0)
            test_dataset.x = np.concatenate(test_dataset.x, axis=0)
            model.fit(train_dataset.x[:, None], train_dataset.y)
            train_acc = model.score(train_dataset.x[:, None], train_dataset.y)
            test_acc = model.score(test_dataset.x[:, None], test_dataset.y)
        target_log = pd.read_csv(r'' + path_to_dataset + '/log.csv', header=0, delimiter=";")
        target_train_acc = np.asarray(target_log["acc"])[-1]
        target_test_acc = np.asarray(target_log["val_acc"])[-1]
        result_file.write("{},{},{},{},{}\n".format(
            directory, target_train_acc, target_test_acc, train_acc, test_acc)
        )
        print("{},{},{},{},{}\n".format(
            directory, target_train_acc, target_test_acc, train_acc, test_acc))

        if args.save_predictions:
            if not os.path.exists(args.out_dir+"/predictions"):
                os.mkdir(args.out_dir+"/predictions")
            prediction_file_name = os.path.join(args.out_dir, 'predictions/{}_train'.format(directory))
            prediction_file = open(prediction_file_name, 'wb')
            predictions = model.predict(train_dataset.x)
            pickle.dump(predictions, prediction_file)
            prediction_file_name = os.path.join(args.out_dir, 'predictions/{}_test'.format(directory))
            prediction_file = open(prediction_file_name, 'wb')
            predictions = model.predict(test_dataset.x)
            pickle.dump(predictions, prediction_file)

    result_file.close()


if __name__ == "__main__":
    main()
