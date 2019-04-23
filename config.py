import numpy as np


class initialConfig:
    ## The following parameters correspond to the machine learning
    ## part of the framework.

    # This parameter refers to the number of outer folds that
    # are being used in order for the k-fold cross-validation
    # to take place.
    kfold_parameter = 5
    kfold_inner_parameter = 4

    n_jobs = 2
    test_dataset = './datasets/dataset-string-similarity-100.csv'

    # the classification method used: basic, basic_sorted, lgm
    classification_method = 'lgm'

    # This parameter contains a list of the various classifiers
    # the results of which will be compared in the experiments.
    classifiers = ['SVM', 'Decision Tree', 'Random Forest', 'AdaBoost',
                   'Naive Bayes', 'MLP', 'Gaussian Process', 'Extra Trees']

    # These are the parameters that constitute the search space
    # in our experiments.
    SVM_hyperparameters = [
        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
         'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [300]},
        {'kernel': ['poly'], 'degree': [1, 2, 3, 4], 'gamma': ['scale'],
         'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [300]},
        {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma': ['scale'], 'max_iter': [300]}
    ]
    DecisionTree_hyperparameters = {
        'max_depth': [i for i in range(1, 33)],
        'min_samples_split': list(np.linspace(0.1, 1, 10)),
        'min_samples_leaf': list(np.linspace(0.1, 0.5, 5)),
        'max_features': [i for i in range(1, 10)]
    }
    RandomForest_hyperparameters = {
        'bootstrap': [True, False],
        'max_depth': [10, 20, 30, 40, 50, 60, 100, None],
        'criterion': ['gini', 'entropy'],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        "n_estimators": [250, 500, 1000]
    }
    XGBoost_hyperparameters = {
        "n_estimators": [500, 1000, 3000],
        # 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        # hyperparameters to avoid overfitting
        'eta': list(np.linspace(0.01, 0.2, 10)),  # 'learning_rate'
        'gamma': [0, 1, 5],
        'subsample': [0.8, 0.9, 1],
        'colsample_bytree': list(np.linspace(0.3, 1, 8)),
        'min_child_weight': [1, 5, 10],
    }
    MLP_hyperparameters = {
        'learning_rate_init': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        'max_iter': [300, 500, 1000],
        'solver': ['sgd', 'adam']
    }

    ## The following parameters correspond to the various options
    ## regarding the textual feature extraction phase.

    # This parameter refers to the percentage of term n-grams to be
    # utilized in the feature extraction phase. E.g. when it is
    # set to 0.1, only the top 10% term n-grams will be used.
    top_k_terms_percentage = 0.1

    # This parameter refers to the percentage of character n-grams to be
    # utilized in the feature extraction phase. E.g. when it is
    # set to 0.1, only the top 10% character n-grams will be used.
    top_k_character_ngrams_percentage = 0.1

    # These parameters refer to the size of the character n-grams
    # and term n-grams respectively.
    character_n_gram_size = 3
    term_n_gram_size = 2

    # This parameter refers to the category levels to be predicted.
    # If level is equal to None, the experiments will be run for
    # all category levels, one at a time.
    # level = [1, 2]
    level = [2]
    # level = [1]

    # This parameter refers to the desired numbers of the top k most probable
    # predictions to be taken into account for computing the top-k accuracies.
    k_error = [5, 10]
