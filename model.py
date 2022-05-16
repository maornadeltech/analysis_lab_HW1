from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import pickle
import optuna
import random
import numpy as np

random.seed(42)
np.random.seed(42)


def print_model_metrics(y, y_pred):
    f1 = f1_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    print()
    print(f'f1 score {f1: .2f}')
    print(f'precision: {precision: .2f}')
    print(f'recall: {recall: .2f}')
    print(f'accuracy: {acc: .2f}')
    print()

    return f1


def train_and_eval_model(model, data: dict):
    model.fit(data['train'][0], data['train'][1])

    y_prob = model.predict_proba(data['test'][0])
    y_pred = model.predict(data['test'][0])
    f1 = print_model_metrics(data['test'][1].astype('float'), y_pred)

    return f1


def test_random_forest(params:dict, data: dict):

    model = RandomForestClassifier(n_estimators=300, max_depth=params['depth'], class_weight='balanced')
    f1 = train_and_eval_model(model, data)

    return f1


def test_log_reg(data: dict):
    model = LogisticRegression()
    f1 = train_and_eval_model(model, data)

    return f1


def test_different_mlp(params: dict, data: dict):
    hidden_layer_sizes = (params['layers'], params['hidden dim'])
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                          activation=params['activation'],
                          max_iter=params['epochs'])

    return train_and_eval_model(model, data)


def test_null_models(data: dict):
    y_pred_all_one = np.ones(data['test'][0].shape[0])
    y_pred_all_zeros = np.zeros(data['test'][0].shape[0])

    y = data['test'][1]
    res = {'one f1': 0, 'zero f1': 0}
    print('results for all ones:')
    res['one f1'] = print_model_metrics(y, y_pred_all_one)
    print('results for all zeros:')
    res['zero f1'] = print_model_metrics(y, y_pred_all_zeros)
    return res


def find_best_mlp(trial, data: dict):
    params = {
        'activation': trial.suggest_categorical("activation", ["relu", "tanh"]),
        'layers': trial.suggest_int("layers", 3, 6),
        'hidden dim': trial.suggest_int("hidden dim", 300, 500),
        'epochs': 1000
    }

    f1 = test_different_mlp(params, data)

    return f1


def find_best_random_forest(trial, data: dict):
    params = {'depth': trial.suggest_int("depth", 5, 15)}
    f1 = test_random_forest(params, data)

    return f1


def find_best_model(data: dict):
    study_rnf = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study_rnf.optimize(lambda trial: find_best_random_forest(trial, data), n_trials=10)

    study_mlp = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study_mlp.optimize(lambda trial: find_best_mlp(trial, data), n_trials=10)

    return study_mlp.best_trial, study_rnf.best_trial

