import sys
import os
import pandas as pd
import pickle
import data_loader
model_path = ''
imputer_path = '/home/student/data/imputer.pkl'

data_path = sys.argv[1]
model = pickle.load(open(model_path, 'rb'))
imputer = pickle.load(open(imputer_path, 'rb'))
res = {'Ids': [], 'SepsisLabel': []}

for path in os.listdir(f'{data_path}'):
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
    p_vec = data_loader.get_patient_vec(p_df, idx)
    res['Ids'].append(p_id)
    res['SepsisLabel'].append(model.predict(vec))

pd.DataFrame.from_dict(res).to_csv('prediction.csv', index=False)
