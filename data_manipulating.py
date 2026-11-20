import pandas as pd
import numpy as np

COLUMNS_TO_DELETE={
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Timestamp",
    "Subflow Fwd Bytes",
}

def clean_data(CSV_PATH):
    data=pd.read_csv(CSV_PATH)

    data.columns=data.columns.str.strip()

    data = data.replace([np.inf,-np.inf],np.nan)
    data = data.dropna()

    data = data.drop(columns=COLUMNS_TO_DELETE, errors="ignore")

    n_before = len(data)
    data = data.drop_duplicates()
    print(f"Usunięto {n_before - len(data)} duplikatów")

    data = data.loc[:,data.nunique()>1]

    data['Label']=data["Label"].apply(lambda x: 0 if x=='BENIGN' else 1)

    x=data.drop('Label', axis=1)
    y=data['Label']

    return x,y

def split_data(x,y, test_size, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(x))
    rng.shuffle(indices)

    n_test = int(round(len(x) * test_size))
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]

def normalize(x_training,x_test):
    mean = x_training.mean(axis=0)
    std = x_training.std(axis=0)
    std[std == 0] = 1
    return (x_training - mean) / std, (x_test - mean) / std