EXPLANATIONS = ["gradient", "guided_backprop", "lrp.epsilon", "integrated_gradients", 'deep_lift', 'smoothgrad', "none"]
ARCHITECTURES = ['purchase_base', 'attack_base', 'texas_base', 'cifar_base', 'adult_base', 'hospital_base']
DATASETS = ['purchase', 'adult', 'texas', 'cifar_100', 'hospital', "fishdog", "cifar_10"]
DIMENSIONS = {"adult": 104, "hospital": 127, "fishdog": 2048}
NEW_POINT_FUNCTIONS = ["lstsq", "exact"]
INFLUENCE_DATASETS = ["adult", "hospital", "fishdog"]
INFLUENCE_MODELTYPES = ["logistic_regression", "svm"]
DATA_PATH = 'data/'
