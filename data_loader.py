import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tqdm import tqdm
import pickle
import os
import random


random.seed(42)
np.random.seed(42)
DATA_PATH = '/home/student/data'


def get_patient_vec(patient_df: pd.DataFrame, idx, num_rows=25):
    patient_df = patient_df.iloc[max(idx - num_rows, 0):idx]

    const = np.array([patient_df.loc[max(idx - num_rows, 0), 'Age'], patient_df.loc[max(idx - num_rows, 0), 'Gender']])
    patient_df = patient_df.drop(['Age', 'Gender'], axis=1)

    mean, std = patient_df.mean(axis=0).values, patient_df.std(ddof=0, axis=0).values
    if patient_df.shape[0] == 1:
        patient_df = pd.concat([patient_df, patient_df])
    trend = np.vstack([sm.tsa.seasonal_decompose(patient_df.loc[:, [col]], model='additive', period=1, extrapolate_trend=1).trend for col in patient_df.columns]).T
    obs_min, obs_max = trend.min(axis=0), trend.max(axis=0)
    obs_first, obs_last, obs_pre = trend[0], trend[-1], trend[len(trend) - 2]
    obs_diff = obs_last - obs_first
    obs_pre_diff = obs_last - obs_pre
    obs_vec = np.vstack((obs_first, obs_last, obs_diff, mean, obs_max, obs_min, std, obs_pre_diff)).flatten('F')

    return np.concatenate([obs_vec, const])


def data_preprocess(num_rows=25):
    if os.path.isfile(f'{DATA_PATH}/my_data.pkl'):  # if given configuration of data exist - return it.
        print('using existing data')
        return pickle.load(open(f'{DATA_PATH}/my_data.pkl', 'rb'))

    train_paths = sorted([f'{DATA_PATH}/train/{path}' for path in os.listdir(f'{DATA_PATH}/train')])
    test_paths = sorted([f'{DATA_PATH}/test/{path}' for path in os.listdir(f'{DATA_PATH}/test')])

    if os.path.isfile(f'{DATA_PATH}/imputer.pkl'):
        imputer = pickle.load(open(f'{DATA_PATH}/imputer.pkl', 'rb'))
    else:
        full_data = pd.concat(
            [pd.read_csv(path, sep='|').interpolate(limit_direction='both', axis=0) for path in test_paths]).drop(
            ['SepsisLabel', 'Unit1', 'Unit2', 'ICULOS'], axis=1)
        imputer = IterativeImputer(random_state=0)
        imputer.fit(full_data.values)
        pickle.dump(imputer, open(f'{DATA_PATH}/imputer.pkl', 'wb'))

    data = {}
    for mode, paths in zip(['train', 'test'], [train_paths, test_paths]):
        vector_list, labels_list, ids_list = [], [], []
        for path in tqdm(paths):
            p_id = path.split("/")[-1].split(".")[-2].split('_')[-1]
            p_df = pd.read_csv(path, sep='|').interpolate(limit_direction='both', axis=0)
            label = p_df['SepsisLabel'].max()
            idx = 0
            if label:
                idx = p_df['SepsisLabel'].values.argmax() + 1
            else:
                idx = p_df.shape[0]
            p_df = p_df.drop(['SepsisLabel', 'Unit1', 'Unit2', 'ICULOS'], axis=1)
            values = imputer.transform(p_df.values)
            p_df = pd.DataFrame(values.round(2), columns=p_df.columns)
            p_vec = get_patient_vec(p_df, idx)
            vector_list.append(p_vec)
            labels_list.append(label)
            ids_list.append(p_id)

        data[mode] = (np.array(vector_list), np.array(labels_list), np.array(ids_list))

    pickle.dump(data, open(f'{DATA_PATH}/my_data.pkl', 'wb'))
    return data





