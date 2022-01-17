import argparse
from datasets import get_dataset
from architectures import get_architecture
from keras import optimizers
from keras.callbacks import CSVLogger
import os
import numpy as np
import innvestigate
import innvestigate.utils as iutils
from time import time
import datetime
# noinspection PyUnresolvedReferences
import setGPU

parser = argparse.ArgumentParser(description='Train target model')
parser.add_argument('-seed', type=int)
args = parser.parse_args()


def main():
    starting_time = time()
    experiment_name = "SmoothGradTest"
    seed = args.seed
    out_dir = "target_output/{}/{}".format(experiment_name, seed)
    dataset = "purchase"
    explanation = "smoothgrad"
    arch = "purchase_base"
    lr = 0.01
    lr_decay = 1e-7
    epochs = 25
    dropout = 0.0
    verbose = 2
    batch = 512

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not seed:
        seed = np.random.randint(low=0, high=1000000)

    print("Creating target dataset")
    train_dataset, test_dataset = get_dataset(dataset, int(seed))

    # Train the model
    print("Training the model")
    model = get_architecture(arch)
    set_dropout(model, dropout)
    log_file_name = os.path.join(out_dir, 'log.csv')
    csv_logger = CSVLogger(log_file_name, append=True, separator=';')

    optimizer = optimizers.Adagrad(lr=lr, decay=lr_decay)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(train_dataset.x, train_dataset.y, epochs=epochs,
              validation_data=(test_dataset.x, test_dataset.y), callbacks=[csv_logger], verbose=verbose)

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

    for number_samples in [-1, 0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 64]:
        print("Number samples : {}".format(number_samples))
        dataset_file_name = os.path.join(out_dir, 'out_dataset_{}.csv'.format(number_samples))
        dataset_file = open(dataset_file_name, 'w')
        if number_samples >= 0:
            dataset_file_name2 = os.path.join(out_dir, 'out_dataset_norm_{}.csv'.format(number_samples))
            dataset_file2 = open(dataset_file_name2, 'w')
            dataset_file.write("y")
            dataset_file2.write("y,norm\n")
            if number_samples == 0:
                analyzer = innvestigate.create_analyzer("gradient", model_wo_softmax)
            else:
                analyzer = innvestigate.create_analyzer(explanation, model_wo_softmax, augment_by_n=number_samples)
            for i in range(len(train_dataset.x[0].reshape(-1))):
                dataset_file.write(",{}_{}".format(explanation, i))
            dataset_file.write("\n")
            for y, dataset in zip([0, 1], [train_dataset, test_dataset]):
                for i in range(0, len(dataset.x), batch):
                    analysis = analyzer.analyze(dataset.x[i:i+batch])
                    for j in range(len(analysis)):
                        dataset_file.write("{}".format(y))
                        dataset_file2.write("{}".format(y))
                        dataset_file2.write(",{}".format(np.linalg.norm(analysis[j], ord=1)))
                        for item in analysis[j].reshape(-1):
                            dataset_file.write(",{:.3E}".format(item))
                        dataset_file.write("\n")
                        dataset_file2.write("\n")
            dataset_file2.close()
        else:

            dataset_file.write("y,label,predicted_label")

            for i in range(len(train_dataset.y[0])):
                dataset_file.write(",prediction_{}".format(i))
            dataset_file.write("\n")
            for y, dataset in zip([0, 1], [train_dataset, test_dataset]):
                for i in range(0, len(dataset.x), batch):
                    predictions = model.predict(dataset.x[i:i + batch])
                    predicted_label = predictions.argmax(axis=1)
                    for j in range(len(predicted_label)):
                        dataset_file.write("{},{},{}".format(y, np.argmax(dataset.y[i + j]), predicted_label[j]))
                        for item in predictions[j]:
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
