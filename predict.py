import sys
import os
import pandas as pd
import pickle
import data_loader
import numpy as np
import tqdm
from sklearn.metrics import f1_score

model_path = 'model.pkl'
imputer_path = 'imputer.pkl'
batch_size = 1000

data_path = sys.argv[1]
model = pickle.load(open(model_path, 'rb'))
imputer = pickle.load(open(imputer_path, 'rb'))
res = {'Ids': [], 'SepsisLabel': []}
vecs = []
reals = []

for path in tqdm.tqdm(os.listdir(data_path)):
    p_id = path.split("/")[-1].split(".")[-2].split('_')[-1]
    p_df = pd.read_csv(f"{data_path}/{path}", sep='|').interpolate(limit_direction='both', axis=0)
    label = p_df['SepsisLabel'].max()
    idx = 0
    if label:
        idx = p_df['SepsisLabel'].values.argmax() + 1
    else:
        idx = p_df.shape[0]
    p_df = p_df.drop(['SepsisLabel', 'Unit1', 'Unit2', 'ICULOS'], axis=1)
    values = imputer.transform(p_df.values)
    p_df = pd.DataFrame(values.round(2), columns=p_df.columns)
    vecs.append(data_loader.get_patient_vec(p_df, idx))
    if len(vecs) == batch_size:
        res['SepsisLabel'].extend(model.predict(np.stack(vecs)).tolist())
        vecs = []
    res['Ids'].append(p_id)
    reals.append(label)

if vecs:
    res['SepsisLabel'].extend(model.predict(np.stack(vecs)).tolist())

print(f"f1 = {f1_score(np.asarray(reals), np.asarray(res['SepsisLabel']))}")

pd.DataFrame.from_dict(res).to_csv('prediction.csv', index=False)
