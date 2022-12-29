import pandas as pd 
import yaml
from sklearn.model_selection import StratifiedKFold
from addict import Dict
import os 


def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)


path_yaml = "./brast2020.yaml"
cfg = read_yaml(path_yaml)

class GlobalConfig():
    root = cfg.Data.root
    train_path = cfg.Data.train_path
    val_path = cfg.Data.val_path
    name_mapping_path = cfg.Data.name_mapping_path
    survival_info_path = cfg.Data.survival_info_path
    train_df = cfg.Data.train_df
    seed = 55

def Fold_df_traning(number_split=7):
   
    name_mapping = pd.read_csv(os.path.join(config.root, config.train_path + config.name_mapping_path))
    survival_info = pd.read_csv(os.path.join(config.root, config.train_path + config.survival_info_path))
    name_mapping.rename({'BraTS_2020_subject_ID': 'Brats20ID'}, axis = 1, inplace = True)
    df = survival_info.merge(name_mapping, on='Brats20ID', how='right')
    path = []
    for _, row in df.iterrows():
        id_ = row['Brats20ID']
        phase = id_.split('_')[-2]
        if phase == 'Training':
            data_path = os.path.join(config.root, config.train_path + id_)
        else:
            data_path = os.path.join(config.root, config.train_path + id_)
        path.append(data_path)
    df['Path'] = path
    df['Age_rank'] = df['Age'].values//10*10
    df= df.loc[df['Brats20ID'] != 'BraTS20_Training_355'].reset_index(drop = True)
    train_df = df.loc[df['Age'].isnull() != True].reset_index(drop = True)
    skf = StratifiedKFold(n_splits=number_split, random_state=config.seed, shuffle = True)
    for i, (train_index, val_index) in enumerate(skf.split(train_df, train_df['Age_rank'])):
        train_df.loc[val_index,['Fold']] = i
    train_data = train_df.loc[train_df['Fold'] != 0.0].reset_index(drop=True)
    val_data = train_df.loc[train_df['Fold'] == 0.0].reset_index(drop=True)
    test_df = df.loc[df['Age'].isnull()].reset_index(drop=True)
    print("train_data ->", train_data.shape, "val_data ->", val_data.shape, "test_df ->", test_df.shape)
    train_df.to_csv(config.train_df, index = False)
    return train_df