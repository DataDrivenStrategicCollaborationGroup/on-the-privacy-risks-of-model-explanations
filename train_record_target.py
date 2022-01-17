from sklearn.linear_model import LogisticRegression
from sklearn import svm
from time import time
from datasets import get_dataset, DATASETS
import datetime
import argparse
import os
import numpy as np
import pickle
from constants import INFLUENCE_MODELTYPES

parser = argparse.ArgumentParser(description='Train target model')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('dataset_size', type=int)
parser.add_argument('model_type', type=str, choices=INFLUENCE_MODELTYPES)
parser.add_argument('out_dir', type=str, help='folder to save model and training log)')
parser.add_argument('--seed', type=int, default=None, help='Random seed for train test split')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
args = parser.parse_args()


def get_model(model_type):
    if model_type == "logistic_regression":
        return LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000)
    elif model_type == "svm":
        return svm.SVC(gamma='scale', kernel='rbf', probability=True)


def test_time(dataset_name, data_size, model_type, number_seeds):
    print(dataset_name, end=", ")
    model = get_model(model_type)
    train_dataset, test_dataset = get_dataset(dataset_name, seed=0, size=data_size, categorical=False)
    starting_time = time()
    print(len(train_dataset.x[0]), end=", ")
    model.fit(train_dataset.x, train_dataset.y)
    print(model.score(train_dataset.x, train_dataset.y), end=", ")
    print(model.score(test_dataset.x, test_dataset.y), end=", ")
    after_time = time()
    time_elapsed = str(datetime.timedelta(seconds=(after_time - starting_time)))
    print(time_elapsed, end=", ")
    estimated_time = str(datetime.timedelta(seconds=number_seeds*data_size*(after_time - starting_time)))
    print(estimated_time)


def train_targets(dataset_name, data_size, model_type, seed, out_dir):
    starting_time = time()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    print("Creating target dataset")
    print(seed)
    train_dataset, test_dataset = get_dataset(dataset_name, seed=seed, size=data_size, categorical=False)
    model = get_model(model_type)
    model.fit(train_dataset.x, train_dataset.y)
    print("{},{:.2f},{:.2f}".format(-1, model.score(train_dataset.x, train_dataset.y),
                                    model.score(test_dataset.x, test_dataset.y)))
    pickle.dump(model, open(out_dir+"/"+str(-1), "wb"))

    for i in range(len(train_dataset.x)):
        leave_i_out_x = np.concatenate([train_dataset.x[:i], train_dataset.x[i + 1:]])
        leave_i_out_y = np.concatenate([train_dataset.y[:i], train_dataset.y[i + 1:]])
        leave_i_out_model = get_model(model_type)
        leave_i_out_model.fit(leave_i_out_x, leave_i_out_y)
        pickle.dump(leave_i_out_model, open(out_dir + "/" + str(i), "wb"))
        after_time = time()
        time_elapsed = str(datetime.timedelta(seconds=(after_time - starting_time)))
        print("{},{:.2f},{:.2f},{}".format(i, leave_i_out_model.score(train_dataset.x, train_dataset.y),
                                           leave_i_out_model.score(test_dataset.x, test_dataset.y), time_elapsed))


def main():
    train_targets(args.dataset, args.dataset_size, args.model_type, args.seed, args.out_dir)
    '''
    data_size = 2000
    number_seeds = 10
    model_type = "logistic_regression" # "svm" #
    for dataset_name in ["fishdog"] : #["adult", "hospital"]:
        test_time(dataset_name, data_size, model_type, number_seeds)
    '''


if __name__ == "__main__":
    main()
