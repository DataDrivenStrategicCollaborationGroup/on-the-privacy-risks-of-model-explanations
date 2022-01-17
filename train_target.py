# This file trains a target network and creates a dataset on which the attack can be trained.
# Written by Martin Strobel
import argparse
from datasets import get_dataset, DATASETS
from architectures import ARCHITECTURES, get_architecture
from keras import optimizers
from keras.callbacks import CSVLogger
import os
import numpy as np
import innvestigate
import innvestigate.utils as iutils
from time import time
import datetime
from constants import EXPLANATIONS
# noinspection PyUnresolvedReferences
import setGPU

parser = argparse.ArgumentParser(description='Train target model')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('out_dir', type=str, help='folder to save model and training log)')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--batch', default=512, type=int, help='batchsize (default: 512)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--lr_decay', type=float, default=1e-7, help='LR decay.')
parser.add_argument('--dropout', type=float, default=0, help='Dropout used in all dropout layers')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=None, help='Random seed for train test split')
parser.add_argument('--print-freq', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--explanation', default="gradient", type=str, help='Explanation used', choices=EXPLANATIONS)
parser.add_argument('--verbose', default=2, type=int, help='Output of training', choices=[0, 1, 2])
args = parser.parse_args()


def main():
    starting_time = time()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if not args.seed:
        args.seed = np.random.randint(low=0, high=1000000)

    print("Creating target dataset")
    train_dataset, test_dataset = get_dataset(args.dataset, int(args.seed))

    # Train the model
    print("Training the model")
    model = get_architecture(args.arch)
    set_dropout(model, args.dropout)
    log_file_name = os.path.join(args.out_dir, 'log.csv')
    csv_logger = CSVLogger(log_file_name, append=True, separator=';')

    if args.dataset == "cifar_10":
        optimizer = optimizers.adam(lr=args.lr)
    else:
        optimizer = optimizers.Adagrad(lr=args.lr, decay=args.lr_decay)
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(train_dataset.x, train_dataset.y, epochs=args.epochs,
              validation_data=(test_dataset.x, test_dataset.y), callbacks=[csv_logger], verbose=args.verbose)

    # Create predictions and explanations
    print("Start creating attacker dataset")
    for layer in model.layers:
        layer.name = layer.name + "FIX_Error"
    # noinspection PyBroadException
    try:
        # noinspection PyUnresolvedReferences
        model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)
    except Exception:
        model_wo_softmax = model

    analyzer = innvestigate.create_analyzer(args.explanation, model_wo_softmax)
    dataset_file_name = os.path.join(args.out_dir, 'out_dataset.csv')
    dataset_file = open(dataset_file_name, 'w')
    dataset_file.write("y,label,predicted_label")
    for i in range(len(train_dataset.y[0])):
        dataset_file.write(",prediction_{}".format(i))
    for i in range(len(train_dataset.x[0].reshape(-1))):
        dataset_file.write(",{}_{}".format(args.explanation, i))
    dataset_file.write("\n")
    for y, dataset in zip([0, 1], [train_dataset, test_dataset]):
        for i in range(0, len(dataset.x), args.batch):
            predictions = model.predict(dataset.x[i:i+args.batch])
            predicted_label = predictions.argmax(axis=1)
            analysis = analyzer.analyze(dataset.x[i:i+args.batch])
            for j in range(len(predicted_label)):
                dataset_file.write("{},{},{}".format(y, np.argmax(dataset.y[i+j]), predicted_label[j]))
                for item in predictions[j]:
                    dataset_file.write(",{:.3E}".format(item))
                for item in analysis[j].reshape(-1):
                    dataset_file.write(",{:.3E}".format(item))
                dataset_file.write("\n")
    dataset_file.close()
    after_time = time()
    time_elapsed = str(datetime.timedelta(seconds=(after_time - starting_time)))
    print(time_elapsed)


def set_dropout(model, dropout):
    for layer in model.layers:
        if "dropout" in layer.name:
            layer.dropout = dropout


if __name__ == "__main__":
    main()
