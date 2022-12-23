import pandas as pd 
import yaml
from sklearn.model_selection import StratifiedKFold
from addict import Dict
import nibabel as nib
import numpy as np 
from torch.utils.data import Dataset, DataLoader


def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)

"""
Create class function for target path and seed random
"""
path_yaml = "./brast2020.yaml
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
   """
    In this dataset of MICCAI_BraTS2020, it has two data files CSV so we need to
    merge them into one data frame to visualize and remove null data 
    """
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
    return train_df.to_csv(config.train_df, index = False)

class BratsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, phase: str="test", is_resize: bool=False):
        self.df = df
        self.phase = phase
        self.augmentations = get_augmentations(phase)
        self.data_types = ['_flair.nii', '_t1.nii', '_t1ce.nii', '_t2.nii']
        self.is_resize = is_resize
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        id_ = self.df.loc[idx, 'Brats20ID']
        root_path = self.df.loc[self.df['Brats20ID'] == id_]['path'].values[0]
        # load all modalities
        images = []
        for data_type in self.data_types:
            img_path = os.path.join(root_path, id_ + data_type)
            img = self.load_img(img_path)#.transpose(2, 0, 1)
            
            if self.is_resize:
                img = self.resize(img)
    
            img = self.normalize(img)
            images.append(img)
        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))
        
        if self.phase != "test":
            mask_path =  os.path.join(root_path, id_ + "_seg.nii")
            mask = self.load_img(mask_path)
            
            if self.is_resize:
                mask = self.resize(mask)
                mask = np.clip(mask.astype(np.uint8), 0, 1).astype(np.float32)
                mask = np.clip(mask, 0, 1)
            mask = self.preprocess_mask_labels(mask)
    
            augmented = self.augmentations(image=img.astype(np.float32), 
                                           mask=mask.astype(np.float32))
            
            img = augmented['image']
            mask = augmented['mask']
    
        
            return {
                "Id": id_,
                "image": img,
                "mask": mask,
            }
        
        return {
            "Id": id_,
            "image": img,
        }
    
    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data
    
    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)
    
    def resize(self, data: np.ndarray):
        data = resize(data, (78, 120, 120), preserve_range=True)
        return data
    
    def preprocess_mask_labels(self, mask: np.ndarray):

        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1

        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1

        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET])
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

        return mask

def get_augmentations(phase):
    list_transforms = []
    
    list_trfms = Compose(list_transforms)
    return list_trfms


def get_dataloader(
    dataset: torch.utils.data.Dataset,
    path_to_csv: str,
    phase: str,
    fold: int = 0,
    batch_size: int = 1,
    num_workers: int = 4,
):
    '''Returns: dataloader for the model training'''
    df = pd.read_csv(path_to_csv)
    
    train_df = df.loc[df['fold'] != fold].reset_index(drop=True)
    val_df = df.loc[df['fold'] == fold].reset_index(drop=True)

    df = train_df if phase == "train" else val_df
    dataset = dataset(df, phase)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,   
    )
    return dataloader        

    

