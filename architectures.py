import keras
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model
import os
from constants import EXPLANATIONS, ARCHITECTURES

path_to_model_folder = "architectures"


def get_architecture(arch_name: str):
    assert arch_name in ARCHITECTURES, "Architecture not available"
    dataset_name = arch_name.split("_")[0]
    # load model
    path_to_model = path_to_model_folder + "/" + arch_name
    try:
        json_file = open(path_to_model+"/model.json", 'r')

        model_architecture_json = json_file.read()
        json_file.close()
        model = model_from_json(model_architecture_json)
    except OSError:
        print("Architecture doesn't exist yet, trying to create it.")
        if dataset_name == 'purchase':
            if arch_name == 'purchase_base':
                model = create_purchase_base()
            else:
                raise NameError(dataset_name+" doesn't have "+arch_name)
        elif dataset_name == 'texas':
            if arch_name == 'texas_base':
                model = create_texas_base()
            else:
                raise NameError(dataset_name+" doesn't have "+arch_name)
        elif dataset_name == 'cifar':
            if arch_name == 'cifar_base':
                model = create_cifar_base()
            else:
                raise NameError(dataset_name+" doesn't have "+arch_name)
        elif dataset_name == 'cifar_10':
            if arch_name == 'cifar_10_base':
                model = create_cifar_base()
            else:
                raise NameError(dataset_name+" doesn't have "+arch_name)
        elif dataset_name == 'adult':
            if arch_name == 'adult_base':
                model = create_adult_base()
            else:
                raise NameError(dataset_name+" doesn't have "+arch_name)
        elif dataset_name == 'hospital':
            if arch_name == 'hospital_base':
                model = create_adult_base()
            else:
                raise NameError(dataset_name + " doesn't have " + arch_name)
        else:
            raise NameError(dataset_name + " doesn't have any architectures")
        model_json = model.to_json()
        if not os.path.exists(path_to_model):
            os.mkdir(path_to_model)
        with open(path_to_model+"/model.json", "w") as json_file:
            json_file.write(model_json)
    return model


def get_attack_model(arch_name: str, feature_list: list, shape_dict: dict):
    assert arch_name in ARCHITECTURES, "Architecture not available"
    # load model
    if arch_name == 'attack_base':
        model = create_attack_base(feature_list, shape_dict)
    else:
        raise NameError(arch_name+" is not implemented. ")
    return model


def create_purchase_base():
    model = Sequential()
    model.add(Dense(1024, activation='tanh', name='base_dense_1',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer='zeros'))
    model.add(Dropout(0, name='dropout_1'))
    model.add(Dense(512, activation='tanh', name='base_dense_2',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer='zeros'))
    model.add(Dropout(0, name='dropout_2'))
    model.add(Dense(256, activation='tanh', name='base_dense_3',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer='zeros'))
    model.add(Dropout(0, name='dropout_3'))
    model.add(Dense(100, activation='softmax', name='base_dense_4',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer='zeros'))
    return model


def create_adult_base():
    # batch_size = 512
    # epochs = 50
    # learning_rate = 0.01
    model = Sequential()
    model.add(Dense(20, activation='tanh',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer='zeros'))
    model.add(Dense(20, activation='tanh',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer='zeros'))
    model.add(Dense(20, activation='tanh',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer='zeros'))
    model.add(Dense(20, activation='tanh',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer='zeros'))
    model.add(Dense(2, activation='softmax',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer='zeros'))

    return model


def create_hospital_base():
    # batch_size = 512
    # epochs = 50
    # learning_rate = 0.01
    model = Sequential()
    model.add(Dense(1024, activation='tanh', name='base_dense_1',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer='zeros'))
    model.add(Dropout(0, name='dropout_1'))
    model.add(Dense(512, activation='tanh', name='base_dense_2',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer='zeros'))
    model.add(Dropout(0, name='dropout_2'))
    model.add(Dense(256, activation='tanh', name='base_dense_3',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer='zeros'))
    model.add(Dropout(0, name='dropout_3'))
    model.add(Dense(2, activation='softmax', name='base_dense_4',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer='zeros'))
    return model


def create_texas_base():
    # batch_size = 512
    # epochs = 50
    # learning_rate = 0.01
    # lr_decay = 1e-7
    model = Sequential()
    # model.add(Dense(20, activation='tanh',
    #                kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
    #                bias_initializer='zeros'))
    # model.add(Dropout(0, name='dropout_3'))
    # model.add(Dense(100, activation='softmax',
    #                kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
    #               bias_initializer='zeros'))
    model.add(Dense(2048, activation='tanh',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer='zeros'))
    model.add(Dense(1024, activation='tanh',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer='zeros'))
    model.add(Dense(512, activation='tanh',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer='zeros'))
    model.add(Dense(256, activation='tanh',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer='zeros'))
    model.add(Dense(100, activation='softmax',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer='zeros'))
    print("#######################")
    print("#######################")
    print("#######################")
    print("#######################")
    print("#######################")
    print("#######################")
    print("#######################")
    return model


def create_cifar_base():
    # batch_size = 512
    # epochs = 50
    # learning_rate = 0.01
    # lr_decay = 1e-7
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation="tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same', activation="tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='tanh',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer='zeros'))
    model.add(Dense(100, activation='softmax',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer='zeros'))
    return model

def create_cifar_10_base():
    # batch_size = 512
    # epochs = 50
    # learning_rate = 0.01
    # lr_decay = 1e-7
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation="tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same', activation="tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='tanh',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer='zeros'))
    model.add(Dense(10, activation='softmax',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer='zeros'))
    return model


def create_attack_base(feature_list, shape_dict):
    used_layers = []
    inputs = []

    if "label" in feature_list:
        label_input = Input(shape=(shape_dict["label"],), name='label_input')
        x_label = Dense(512, activation='relu',
                        kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                        bias_initializer='zeros')(label_input)
        x_label = Dense(64, activation='relu',
                        kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                        bias_initializer='zeros')(x_label)
        used_layers.append(x_label)
        inputs.append(label_input)

    if "predicted_label" in feature_list:
        predicted_label_input = Input(shape=(shape_dict["predicted_label"],), name='predicted_label_input')
        x_predicted_label = Dense(512, activation='relu',
                                  kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                                  bias_initializer='zeros')(predicted_label_input)
        x_predicted_label = Dense(64, activation='relu',
                                  kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                                  bias_initializer='zeros')(x_predicted_label)
        used_layers.append(x_predicted_label)
        inputs.append(predicted_label_input)

    if "prediction" in feature_list:
        prediction_input = Input(shape=(shape_dict["prediction"],), name='prediction_input')
        x_prediction = Dense(1024, activation='relu',
                             kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                             bias_initializer='zeros')(prediction_input)
        x_prediction = Dense(512, activation='relu',
                             kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                             bias_initializer='zeros')(x_prediction)
        x_prediction = Dense(64, activation='relu',
                             kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                             bias_initializer='zeros')(x_prediction)
        used_layers.append(x_prediction)
        inputs.append(prediction_input)

    explanation = test_explanation_in_feature_list(feature_list)
    if explanation:
        gradient_input = Input(shape=(shape_dict[explanation],), name=str(explanation)+'_input')
        x_gradient = Dense(1024, activation='relu',
                           kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer='zeros')(gradient_input)
        x_gradient = Dense(512, activation='relu',
                           kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer='zeros')(x_gradient)
        x_gradient = Dense(64, activation='relu',
                           kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer='zeros')(x_gradient)
        used_layers.append(x_gradient)
        inputs.append(gradient_input)

    if len(used_layers) >1:
        x1 = keras.layers.concatenate(used_layers)
    else:
        x1 = used_layers[0]

    # We stack a deep densely-connected network on top
    x2 = Dense(256, activation='relu',
               kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
               bias_initializer='zeros')(x1)
    x3 = Dense(64, activation='relu',
               kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
               bias_initializer='zeros')(x2)

    # And finally we add the main logistic regression layer
    main_output = Dense(1, activation='sigmoid', name='output',
                        kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                        bias_initializer='zeros')(x3)
    return Model(inputs=inputs, outputs=[main_output])


def test_explanation_in_feature_list(feature_list):
    for explanation in EXPLANATIONS:
        if explanation in feature_list:
            return explanation
    return False
