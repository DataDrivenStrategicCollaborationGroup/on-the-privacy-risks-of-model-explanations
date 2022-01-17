import numpy as np
from time import time
from datasets import get_dataset
from constants import DIMENSIONS, NEW_POINT_FUNCTIONS, INFLUENCE_DATASETS, INFLUENCE_MODELTYPES
import pickle
import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser(description='Run the influence reduction attack')
parser.add_argument('dataset', type=str, choices=INFLUENCE_DATASETS)
parser.add_argument('data_size', type=int, help='Size of the training set')
parser.add_argument('--model_type', type=str, choices=INFLUENCE_MODELTYPES, default="logistic_regression",
                    help='The model type of the target model')
parser.add_argument('--new_point_function', type=str, choices=NEW_POINT_FUNCTIONS, default="lstsq",
                    help='The function used to create the next point.')
parser.add_argument('--seed_from', default=0, type=int, help='The starting seed used to create the training set')
parser.add_argument('--seed_to', default=10, type=int, help='The ending seed used to create the training set')
args = parser.parse_args()


def get_influence(y, org_model, loo_models, train_x):
    org_predict = org_model.predict_proba(y)[:, 0]
    loo_predict = []
    for i in range(len(loo_models)):
        loo_predict.append(loo_models[i].predict_proba(y)[:, 0])
    loo_predict = np.asarray(loo_predict).T
    influences = loo_predict - org_predict.repeat(len(loo_models)).reshape([len(y), len(loo_models)])
    numbers = np.abs(influences).argmax(axis=1)
    return_influences = np.asarray([influences[i, number] for i, number in enumerate(numbers)])
    return train_x[numbers], return_influences, numbers


def reconstruct_original_model(org_model, n):
    b = np.log((1/org_model.predict_proba(np.zeros([1, n]))[0, 0])-1)
    input_ = np.zeros([n, n]).astype(np.double)
    for i in range(n):
        input_[i, i] = 1
    w = np.log(1/(org_model.predict_proba(input_)[:, 0])-1) - b
    return w, b


def recover_new_subspace(y, basis, w, b, org_model, loo_models, train_x, epsilon=1e-4):
    point, infl, number = get_influence(y, org_model=org_model, loo_models=loo_models, train_x=train_x)
    not_all_points_found = True
    influences = None
    test_points = None
    while not_all_points_found:
        test_points = epsilon*basis+y
        points, influences, numbers = get_influence(test_points, org_model=org_model, loo_models=loo_models,
                                                    train_x=train_x)
        if np.asarray(number == numbers).all():
            not_all_points_found = False
        else:
            print("Epsilon too big.")
            epsilon = epsilon/2
    predict_probability = (infl + 1/(1+np.exp(np.matmul(w, y.T)+b)))
    value = np.log(1/predict_probability-1)
    predict_probabilities = influences + 1/(1+np.exp(np.matmul(w, test_points.T)+b))
    values = np.log(1/predict_probabilities-1)
    coefficient = np.linalg.solve(test_points-y, values-value)
    intercept = value-np.matmul(coefficient, y[0, :])
    if not np.allclose(coefficient, loo_models[number[0]].coef_[0, :]) \
            or not np.allclose(intercept, loo_models[number[0]].intercept_[0]):
        print(np.allclose(coefficient, loo_models[number[0]].coef_[0, :]),
              np.allclose(intercept, loo_models[number[0]].intercept_[0]))
        print(np.max(np.abs(coefficient-loo_models[number[0]].coef_[0, :])))
    return coefficient, intercept


def algorithm_attack(dataset, data_size, dimension,  seed, path, model_type, get_new_point):
    start = time()
    print("######{}####".format(seed))
    train_dataset, test_dataset = get_dataset(dataset, seed=seed, size=data_size, categorical=False)
    loo_models = []
    for i in range(data_size):
        loo_models.append(pickle.load(open(path.format(dataset, model_type, seed, i), "rb")))
    org_model = pickle.load(open(path.format(dataset, model_type, seed, -1), "rb"))
    numbers, coefficients, intercepts = [], [], []
    basis = np.zeros([dimension, dimension])
    for i in range(dimension):
        basis[i, i] = 1
    y = np.zeros([1, dimension])
    point, infl, number = get_influence(y, org_model=org_model, loo_models=loo_models, train_x=train_dataset.x)
    numbers.append(number)
    w, b = reconstruct_original_model(org_model, dimension)
    coefficients.append(w)
    intercepts.append(b)
    for i in range(dimension):
        end = time()
        print("{}, {}, {:.2f}".format(i, len(np.unique(numbers)), end - start))
        coefficient, intercept = recover_new_subspace(y.astype(np.double), basis.astype(np.double),
                                                      w, b, org_model=org_model, loo_models=loo_models,
                                                      train_x=train_dataset.x, epsilon=1e-4)
        coefficients.append(coefficient)
        intercepts.append(intercept)
        y = get_new_point(w, b, coefficients, intercepts)
        if type(y) == int:
            break
        _, _, number = get_influence(y, org_model=org_model, loo_models=loo_models, train_x=train_dataset.x)
        numbers.append(number)
    return len(np.unique(numbers))


def get_new_point_lstsq(w, b, coefficients, intercepts):
    if len(coefficients) > np.linalg.matrix_rank(np.asarray(coefficients)):
        print("Singular")
        return -1
    y = np.zeros([1, len(w)])
    found_points = len(intercepts)
    matrix = np.ones([found_points, len(w)]) * w
    bias = -b * np.ones(found_points)
    for i in range(1, found_points):
        bias[i] += intercepts[i]
        matrix[i] -= coefficients[i]
    bias[0] = bias[0] / 2

    y[0] = np.linalg.lstsq(matrix, bias, rcond=None)[0]
    if np.abs(y[0]).max() > 10 ** 3:
        print("Warning!")
    return y


def get_new_point_exact(w, b, coefficients, intercepts):
    ys = []
    counter = 0
    found_points = len(intercepts)
    bias = -b * np.ones(found_points)
    matrix = np.ones([found_points, len(w)]) * w
    bias[0] = bias[0] / 2
    for i in range(1, found_points):
        bias[i] += intercepts[i]
        matrix[i] -= coefficients[i]
    rank = np.linalg.matrix_rank(matrix)
    if len(coefficients) > rank:
        print("Singular")
        return -1
    while counter < 1e8:
        permutation = np.random.permutation(len(w))
        selection = 1 == np.ones(len(w))
        for i in permutation:
            selection[i] = False
            sub_rank = np.linalg.matrix_rank(matrix[:, selection])
            if sub_rank < rank:
                selection[i] = True
            if selection.sum() == rank:
                break
        y = np.zeros([1, len(w)])
        y[0, selection] = np.linalg.solve(matrix[:, selection], bias)
        if np.abs(y[0]).max() > 10 ** 2:
            ys.append(y)
            if np.abs(np.asarray(ys).mean(axis=0)).max() < 5 * 10 ** 2:
                if counter > 50:
                    print("Tries: {}".format(counter))
                return np.asarray(ys).mean(axis=0)
            counter += 1
        else:
            if counter > 50:
                print("Tries: {}".format(counter))
            return y
    return -1


new_point_functions = {"exact": get_new_point_exact, "lstsq": get_new_point_lstsq}


def main():
    path = "record_output/{}_{}/{}/{}"
    dimension = DIMENSIONS[args.dataset]
    result = []
    for seed in range(args.seed_from, args.seed_to):
        result.append(algorithm_attack(args.dataset, args.data_size, dimension, seed, path, args.model_type,
                      new_point_functions[args.new_point_function]))
    df = pd.DataFrame()
    df[args.dataset] = np.asarray(result)
    if not os.path.exists('record_output/algorithm1'):
        os.mkdir('record_output/algorithm1')
    df.to_csv("{}/{}/{}.csv".format('record_output', 'algorithm1', args.dataset), sep="\t", index=False)


if __name__ == "__main__":
    main()
