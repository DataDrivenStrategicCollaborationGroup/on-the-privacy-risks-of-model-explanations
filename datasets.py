import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.datasets import cifar100, cifar10
import tensorflow as tf
from tensorflow.python.keras import backend as k
from collections import namedtuple
from constants import DATASETS
import pickle
from constants import DATA_PATH


class Dataset:

    def __init__(self, x, y):
        self.x = x
        self.y = y


def get_dataset(dataset: str, seed: int, size=10000, categorical=True):
    assert dataset in DATASETS, "This dataset is not available"
    if dataset == 'purchase':
        return create_purchase(seed, size, categorical)
    elif dataset == 'adult':
        return create_adult(seed, size, categorical)
    elif dataset == 'texas':
        return create_texas(seed, size, categorical)
    elif dataset == 'cifar_100':
        return create_cifar_100(seed, size, categorical)
    elif dataset == 'cifar_10':
        return create_cifar_100(seed, size, categorical)
    elif dataset == 'hospital':
        return create_hospital(seed, size, categorical)
    elif dataset == 'fishdog':
        return create_fish_dog(seed, categorical)
    else:
        raise NameError("This dataset is not available")


def create_attack_dataset(path_to_directory: str, used_features: list, seed, one_hot_encode_labels=True,
                          one_norm_features=None, variance_features=None, normalize_gradient=False):
    if one_norm_features is None:
        one_norm_features = []
    if variance_features is None:
        variance_features = []
    data = pd.read_csv(r'' + path_to_directory + '/out_dataset.csv', header=0)
    feature_columns = {}
    for feature in used_features:
        feature_columns[feature] = []
        if feature == "label":
            feature_columns[feature].append(feature)
            continue
        if feature == "loss":
            for column in data.columns:
                if "prediction" in column:
                    feature_columns[feature].append(column)
            feature_columns[feature].append("label")
        for column in data.columns:
            if feature in column:
                feature_columns[feature].append(column)
    ind_train, ind_test, y_train, y_test = train_test_split(np.arange(len(data["y"])), data["y"],
                                                            test_size=0.30, random_state=seed)
    x_train, x_test = [], []
    for feature in used_features:
        feature_train_data = np.asarray(data[feature_columns[feature]])[ind_train]
        feature_test_data = np.asarray(data[feature_columns[feature]])[ind_test]
        if one_hot_encode_labels and len(feature_train_data[0]) == 1:
            num_cat = max(np.max(feature_train_data), np.max(feature_test_data)) + 1
            feature_train_data = keras.utils.to_categorical(feature_train_data, num_cat)
            feature_test_data = keras.utils.to_categorical(feature_test_data, num_cat)
        if feature in one_norm_features:
            feature_train_data = np.linalg.norm(feature_train_data, ord=1, axis=1)
            feature_test_data = np.linalg.norm(feature_test_data, ord=1, axis=1)
        if feature in variance_features:
            feature_train_data = np.var(feature_train_data, axis=1)
            feature_test_data = np.var(feature_test_data, axis=1)
        if normalize_gradient and feature == "gradient":
            feature_train_data = feature_train_data/np.linalg.norm(feature_train_data, ord=1, axis=1)[:, None]
            feature_test_data = feature_test_data/np.linalg.norm(feature_test_data, ord=1, axis=1)[:, None]
        if feature == "loss":
            num_cat = max(np.max(feature_train_data[:, -1]), np.max(feature_test_data[:, -1])) + 1
            feature_train_data = calculate_loss(feature_train_data, int(num_cat))
            feature_test_data = calculate_loss(feature_test_data, int(num_cat))
        x_train.append(feature_train_data)
        x_test.append(feature_test_data)
    return Dataset(x=x_train, y=y_train), Dataset(x=x_test, y=y_test)


def calculate_loss(feature_data, num_cat):
    label = keras.utils.to_categorical(feature_data[:, -1].astype(int), num_cat)
    prediction = feature_data[:, :-1]
    sess = k.get_session()
    return keras.losses.categorical_crossentropy(tf.cast(tf.convert_to_tensor(label), tf.float64),
                                                 tf.convert_to_tensor(prediction)).eval(session=sess)


def create_purchase(seed, data_size, categorical):
    data = pd.read_csv(r'' + DATA_PATH + 'purchase/purchase.csv', header=None)
    y = np.asarray(data[0])
    x = np.asarray(data)[:, 1:]
    np.random.seed(seed)
    ind = np.random.choice(range(len(y)), size=2 * data_size, replace=False)
    x_train, x_test, y_train, y_test = train_test_split(x[ind], y[ind], test_size=0.5, random_state=42)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = make_categorical(y_train-1, categorical, 100)
    y_test = make_categorical(y_test-1, categorical, 100)
    return Dataset(x=x_train, y=y_train), Dataset(x=x_test, y=y_test)


def create_texas(seed, data_size, categorical):
    data = pd.read_csv(r'' + DATA_PATH + '/texas/100/feats', header=None)
    labels = pd.read_csv(r'' + DATA_PATH + '/texas/100/labels', header=None)
    y = np.asarray(labels) -1
    x = np.asarray(data)
    np.random.seed(seed)
    ind = np.random.choice(range(len(y)), size=2 * data_size, replace=False)
    x_train, x_test, y_train, y_test = train_test_split(x[ind], y[ind], test_size=0.5, random_state=42)
    #x_train = x_train.astype('float32')
    #x_test = x_test.astype('float32')

    y_train = make_categorical(y_train, categorical, 100)
    y_test = make_categorical(y_test, categorical, 100)
    return Dataset(x=x_train, y=y_train), Dataset(x=x_test, y=y_test)


def create_cifar_100(seed, data_size, categorical):
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    y = np.asarray(np.concatenate(([y_train, y_test])))
    x = np.asarray(np.concatenate(([x_train, x_test])))
    np.random.seed(seed)
    ind = np.random.choice(range(len(y)), size=2 * data_size, replace=False)
    x_train, x_test, y_train, y_test = train_test_split(x[ind], y[ind], test_size=0.5, random_state=42)
    y_train = make_categorical(y_train, categorical, 100)
    y_test = make_categorical(y_test, categorical, 100)
    return Dataset(x=x_train, y=y_train), Dataset(x=x_test, y=y_test)

def create_cifar_10(seed, data_size, categorical):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y = np.asarray(np.concatenate(([y_train, y_test])))
    x = np.asarray(np.concatenate(([x_train, x_test])))
    np.random.seed(seed)
    ind = np.random.choice(range(len(y)), size=2 * data_size, replace=False)
    x_train, x_test, y_train, y_test = train_test_split(x[ind], y[ind], test_size=0.5, random_state=42)
    y_train = make_categorical(y_train, categorical, 10)
    y_test = make_categorical(y_test, categorical, 10)
    return Dataset(x=x_train, y=y_train), Dataset(x=x_test, y=y_test)


def create_adult(seed, data_size, categorical):

    dataset_train, dataset_test, _ = load_adult(DATA_PATH + "/adult/adult.data", DATA_PATH + "/adult/adult.test")
    y = np.asarray(np.concatenate(([dataset_train.target, dataset_test.target])))
    x = np.asarray(np.concatenate(([dataset_train.data, dataset_test.data])))
    np.random.seed(seed)
    ind = np.random.choice(range(len(y)), size=2 * data_size, replace=False)
    x_train, x_test, y_train, y_test = train_test_split(x[ind], y[ind], test_size=0.5, random_state=42)
    y_train = make_categorical(y_train, categorical, 2)
    y_test = make_categorical(y_test, categorical, 2)

    return Dataset(x=x_train, y=y_train), Dataset(x=x_test, y=y_test)


def create_hospital(seed, data_size, categorical):

    x, y = load_hospital()
    y = (y + 1) / 2

    np.random.seed(seed)
    num_examples = len(y)
    assert x.shape[0] == num_examples
    num_train_examples_per_class = int(data_size / 2)
    num_test_examples_per_class = int(data_size / 2)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    assert len(pos_idx) + len(neg_idx) == num_examples

    train_idx = np.concatenate((pos_idx[:num_train_examples_per_class], neg_idx[:num_train_examples_per_class]))
    test_idx = np.concatenate((pos_idx[num_train_examples_per_class:num_train_examples_per_class +
                               num_test_examples_per_class],
                               neg_idx[num_train_examples_per_class:num_train_examples_per_class +
                               num_test_examples_per_class]))
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    x_train = np.array(x.iloc[train_idx, :], dtype=np.float32)
    y_train = y[train_idx]

    x_test = np.array(x.iloc[test_idx, :], dtype=np.float32)
    y_test = y[test_idx]

    y_train = make_categorical(y_train, categorical, 2)
    y_test = make_categorical(y_test, categorical, 2)

    return Dataset(x=x_train, y=y_train), Dataset(x=x_test, y=y_test)


def create_fish_dog(seed, categorical):

    x = pickle.load(open(DATA_PATH + "fishdog/latent/x.p", "rb"))
    y = pickle.load(open(DATA_PATH + "fishdog/latent/y.p", "rb"))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=seed)
    y_train = make_categorical(y_train, categorical, 100)
    y_test = make_categorical(y_test, categorical, 100)
    return Dataset(x=x_train, y=y_train), Dataset(x=x_test, y=y_test)


def make_categorical(y, categorical, num_classes):
    if categorical:
        return keras.utils.to_categorical(y, num_classes)
    else:
        return y.astype(int)


# credits: http://stackoverflow.com/questions/2356925/how-to-check-whether-string-might-be-type-cast-to-float-in-python
def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def find_means_for_continuous_types(x):
    means = []
    for col in range(len(x[0])):
        m_sum = 0
        count = 0.000000000000000000001
        for value in x[:, col]:
            if is_float(value):
                m_sum += float(value)
                count += 1
        means.append(m_sum / count)
    return means


def prepare_data(raw_data, means, inputs, input_shape):
    x = raw_data[:, :-1]
    y = raw_data[:, -1:]

    # x:
    def flatten_persons_inputs_for_model(person_inputs, j_means, j_inputs, j_input_shape):
        float_inputs = []
        for l in range(len(j_input_shape)):
            features_of_this_type = j_input_shape[l]
            is_feature_continuous = j_inputs[l][1][0] == 'continuous'

            if is_feature_continuous:
                mean = j_means[l]
                if is_float(person_inputs[l]):
                    scale_factor = 1 / (2 * mean)  # we prefer inputs mainly scaled from -1 to 1.
                    float_inputs.append(float(person_inputs[l]) * scale_factor)
                else:
                    float_inputs.append(mean)
            else:
                for j in range(features_of_this_type):
                    feature_name = j_inputs[l][1][j]
                    if feature_name == person_inputs[l]:
                        float_inputs.append(1.)
                    else:
                        float_inputs.append(0)
        return float_inputs

    new_x = []
    for person in range(len(x)):
        formatted_x = flatten_persons_inputs_for_model(x[person], means, inputs, input_shape)
        new_x.append(formatted_x)
    new_x = np.array(new_x)

    # y:
    new_y = []
    for i in range(len(y)):
        if y[i] == ">50K":
            new_y.append(1)
        else:  # y[i] == "<=50k":
            new_y.append(0)
    new_y = np.array(new_y)
    data = namedtuple('_', 'data, target')(new_x, new_y)
    return data


def load_adult(path_train, path_test):
    inputs = (
        ("age", ("continuous",)),
        ("workclass", ("Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov",
                       "Without-pay", "Never-worked")),
        ("fnlwgt", ("continuous",)),
        ("education", ("Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th",
                       "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool")),
        ("education-num", ("continuous",)),
        ("marital-status", ("Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
                            "Married-spouse-absent", "Married-AF-spouse")),
        ("occupation", ("Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
                        "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
                        "Priv-house-serv", "Protective-serv", "Armed-Forces")),
        ("relationship", ("Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried")),
        ("race", ("White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black")),
        ("sex", ("Female", "Male")),
        ("capital-gain", ("continuous",)),
        ("capital-loss", ("continuous",)),
        ("hours-per-week", ("continuous",)),
        ("native-country", ("United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
                            "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran",
                            "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal",
                            "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia",
                            "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
                            "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands")),
    )

    input_shape = []
    column_names = []
    for i in inputs:
        count = len(i[1])
        if count <= 2:
            count = 1
            column_names.append(i[0])
        else:
            for j in range(len(i[1])):
                column_names.append("{}_{}".format(i[0], i[1][j]))
        input_shape.append(count)

    cols = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
            "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"]
    re_cols = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
               "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
               "Hours per week", "Country", "Target"]

    training_data = pd.read_csv(
                    path_train,
                    names=cols,
                    sep=r'\s*,\s*',
                    engine='python',
                    na_values="?")
    training_data = training_data[re_cols].values
    test_data = pd.read_csv(
                    path_test,
                    names=cols,
                    sep=r'\s*,\s*',
                    engine='python',
                    na_values="?")
    test_data = test_data[re_cols].values
    means = find_means_for_continuous_types(np.concatenate((training_data, test_data), 0))

    d_train = prepare_data(training_data, means, inputs, input_shape)
    d_test = prepare_data(test_data, means, inputs, input_shape)
    return d_train, d_test, column_names


def load_hospital():
    df = pd.read_csv(DATA_PATH + "/hospital/hospital")

    x = df.loc[:, ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient',
                   'number_emergency', 'number_inpatient', 'number_diagnoses']]

    categorical_var_names = ['gender', 'race', 'age', 'discharge_disposition_id', 'max_glu_serum', 'A1Cresult',
                             'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
                             'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone',
                             'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin',
                             'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
                             'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed']

    for categorical_var_name in categorical_var_names:
        categorical_var = pd.Categorical(
            df.loc[:, categorical_var_name])
        # Just have one dummy variable if it's boolean
        if len(categorical_var.categories) == 2:
            drop_first = True
        else:
            drop_first = False

        dummies = pd.get_dummies(
            categorical_var,
            prefix=categorical_var_name,
            drop_first=drop_first)

        x = pd.concat([x, dummies], axis=1)

    # Set the Y labels
    readmitted = pd.Categorical(df.readmitted)
    y = np.copy(readmitted.codes)
    # Combine >30 and 0 and flip labels, so 1 (>30) and 2 (No) become 0, while 0 becomes 1
    y[y >= 1] = -1
    y[y == 0] = 1
    return x, y
