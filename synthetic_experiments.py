import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import pickle
import numpy as np
from sklearn.datasets import make_classification
import innvestigate
from sklearn.model_selection import train_test_split
import innvestigate.utils as iutils
import argparse
import os
# noinspection PyUnresolvedReferences
import setGPU

parser = argparse.ArgumentParser(description='Train target model')
parser.add_argument('experiment', type=str)
parser.add_argument('out_dir', type=str, help='folder to save output')
parser.add_argument('--iteration_from', type=int, metavar='N', default=0,
                    help='iteration from')
parser.add_argument('--iteration_to', type=int, metavar='N', default=1,
                    help='iteration to')

parser.add_argument('--number_classes', default=2, type=int, metavar='N',
                    help='number of  classes in dataset')
parser.add_argument('--n_informative_features', default=2, type=int, metavar='N',
                    help='number of  informative features in network')
parser.add_argument('--number_layers', default=2, type=int, metavar='N',
                    help='number of layers in network')
parser.add_argument('--number_nodes', default=50, type=int, metavar='N',
                    help='number of nodes per layer')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_decay', type=float, default=1e-7,
                    help='LR decay.')
parser.add_argument('--verbose', default=2, type=int, help='Output of training', choices=[0, 1, 2])


args = parser.parse_args()


class Dataset:

    def __init__(self, x, y):
        self.x = x
        self.y = y


def build_model(number_classes=2, number_layers=2, number_nodes=50):
        model = Sequential()
        for layer in range(number_layers):
            model.add(Dense(number_nodes, activation='tanh',
                      kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                      bias_initializer='zeros'))
        model.add(Dense(number_classes, activation='softmax',
                        kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                        bias_initializer='zeros'))
        return model


def train_model(model, train_dataset, test_dataset, max_epochs=100, learning_rate=0.01, lr_decay=1e-7, verbose=0):

        # optimization details
        sgd = optimizers.Adagrad(lr=learning_rate, decay=lr_decay)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        model.fit(train_dataset.x, train_dataset.y,
                  epochs=max_epochs, batch_size=512,
                  validation_data=(test_dataset.x, test_dataset.y), verbose=verbose)
        return model


def main():
    experiment = args.experiment
    iteration_from = args.iteration_from
    iteration_to = args.iteration_to
    number_classes = args.number_classes
    n_informative_features = args.n_informative_features
    number_layers = args.number_layers
    number_nodes = args.number_nodes
    max_epochs = args.epochs
    learning_rate = args.lr
    lr_decay = args.lr_decay
    verbose = args.verbose
    out_dir = args.out_dir
    # informative_feature_numbers = [1, 2, 5, 10, 14, 20, 50, 100, 127, 200, 500, 600, 1000, 3072, 5000, 6000, 10000]
    if not os.path.exists(out_dir + "/" + experiment):
        os.mkdir(out_dir + "/" + experiment)
    for number_iteration in range(iteration_from, iteration_to):
        print("\n#############{}#############".format(number_iteration))
        print("\n {} \n ".format(number_classes), end="")
        if n_informative_features < 20:
            if 2 ** n_informative_features < number_classes:
                continue  # This is required so we can generate the dataset
        print(n_informative_features, end=", ")
        x, y = make_classification(n_samples=20000, n_features=n_informative_features,
                                   n_informative=n_informative_features, n_redundant=0, n_repeated=0,
                                   n_classes=number_classes, n_clusters_per_class=1,
                                   random_state=number_iteration)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=33)
        x_train, x_test = x_train.astype('float32'), x_test.astype('float32')

        y_train = keras.utils.to_categorical(y_train, number_classes)
        y_test = keras.utils.to_categorical(y_test, number_classes)

        train_dataset = Dataset(x_train, y_train)
        test_dataset = Dataset(x_test, y_test)
        model = train_model(build_model(number_classes=number_classes, number_layers=number_layers,
                                        number_nodes=number_nodes),
                            train_dataset, test_dataset, max_epochs=max_epochs, learning_rate=learning_rate,
                            lr_decay=lr_decay, verbose=verbose)

        # noinspection PyBroadException
        try:
            # noinspection PyUnresolvedReferences
            model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)
        except Exception:
            model_wo_softmax = model

        analyzer = innvestigate.create_analyzer('gradient', model_wo_softmax)
        ana_train = analyzer.analyze(x_train)
        ana_test = analyzer.analyze(x_test)
        ana_train_one_norm = np.linalg.norm(ana_train, axis=1, ord=1)
        ana_test_one_norm = np.linalg.norm(ana_test, axis=1, ord=1)
        one_norm = np.concatenate([ana_train_one_norm, ana_test_one_norm])
        membership = np.concatenate([np.ones(len(ana_train_one_norm)), np.zeros(len(ana_test_one_norm))])
        correlation = np.corrcoef(one_norm, membership)[0, 1]
        pickle.dump(correlation, open('{}/{}/correlation_{}_{}_{}.p'.format(out_dir, experiment,
                                                                            number_iteration, number_classes,
                                                                            n_informative_features), 'wb'))
        pickle.dump(model.evaluate(x_test, y_test, verbose=0)[1],
                    open('{}/{}/accuracy_{}_{}_{}.p'.format(out_dir, experiment, number_iteration, number_classes,
                                                            n_informative_features), 'wb'))
        pickle.dump(model.evaluate(x_train, y_train, verbose=0)[1],
                    open('{}/{}/train_accuracy_{}_{}_{}.p'.format(out_dir, experiment, number_iteration, number_classes,
                                                                  n_informative_features), 'wb'))


if __name__ == "__main__":
    main()
