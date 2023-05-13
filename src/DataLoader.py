import pandas as pd 
import nibabel as nib
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from utils import Fold_df_traning
import torchio as tio

class BratsDataSet(Dataset):
    def __init__(self, df: pd.DataFrame, phase: str = 'test'):
        self.df = df
        self.phase = phase
        self.data_types = ['_flair.nii', '_t1.nii', '_t1ce.nii', '_t2.nii']
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, index):
        id_ = self.df.loc[index, 'Brats20ID']
        data_path = self.df.loc[self.df['Brats20ID'] == id_]['Path'].values[0]
        data_img = []
        resample = tio.Resample((2,2,2))
#         normalize = tio.ZNormalization()
        for data_type in self.data_types:
            img = tio.ScalarImage(os.path.join(data_path, id_ + data_type)) #data_img shape (1, 240, 240, 155)
            img = resample(img) #data_img shape (1, 120, 120, 78)
            img = np.array(img)
            img = np.squeeze(img, axis = 0)
            img = self.Normalize(img)
            data_img.append(img)
        img_stack = np.stack(data_img)
        img_stack = np.moveaxis(img_stack, (0,1,2,3), (0,3,2,1))
        img_stack = torch.Tensor(img_stack)
        
        if self.phase != 'test':
            labels = tio.LabelMap(os.path.join(data_path, id_ + '_seg.nii'))
            labels = resample(labels)
            labels = np.array(labels)
            labels = np.squeeze(labels, axis = 0)
            label_stack = self.ConvertToMultiChannel(labels)
            label_stack = torch.Tensor(label_stack)
            
            subjects = tio.Subject(image = tio.ScalarImage(tensor = img_stack),
                                   label = tio.LabelMap(tensor = (label_stack > 0.5)),
                                   id = id_
                                  )
            
            
            return subjects
        subjects = tio.Subject(image = tio.ScalarImage(tensor = img_stack),
                               id = id_
                              )
        return subjects
    
    def Normalize(self, image : np.ndarray):
        return (image - np.min(image))/(np.max(image) - np.min(image))
 
    def ConvertToMultiChannel(self, labels):
        '''
        Convert labels to multi channels based on brats classes:
        label 1 is the peritumoral edema
        label 2 is the GD-enhancing tumor
        label 3 is the necrotic and non-enhancing tumor core
        The possible classes are TC (Tumor core), WT (Whole tumor)
        and ET (Enhancing tumor)
        '''
        label_TC = labels.copy()
        label_TC[label_TC == 1] = 1
        label_TC[label_TC == 2] = 0
        label_TC[label_TC == 4] = 1
        
        
        label_WT = labels.copy()
        label_WT[label_WT == 1] = 1
        label_WT[label_WT == 2] = 1
        label_WT[label_WT == 4] = 1
        
        label_ET = labels.copy()
        label_ET[label_ET == 1] = 0
        label_ET[label_ET == 2] = 0
        label_ET[label_ET == 4] = 1
        
        label_stack = np.stack([label_WT, label_TC, label_ET])
        label_stack = np.moveaxis(label_stack, (0,1,2,3), (0,3,2,1))
        return label_stack

# List transform
def get_agumentation(phase):
    if phase == 'train':
        # As RandomAffine is faster then RandomElasticDeformation, we choose to
        # apply RandomAffine 80% of the times and RandomElasticDeformation the rest
        # Also, there is a 25% chance that none of them will be applied
        list_transforms = [
            tio.Resize((50,50,50)),
            tio.RandomBiasField(p = 0.25),
            tio.RandomBlur(p = 0.25),
            tio.RescaleIntensity(out_min_max=(0, 1)),
            
        ]
        # Transforms can be composed as in torchvision.transforms
        transform = tio.Compose(list_transforms)
    else:
        list_transforms = [
                        tio.RescaleIntensity(out_min_max=(0, 1)),
            
        ]
        transform = tio.Compose(list_transforms)
    return transform   

def get_dataloader(dataset, path_to_csv, phase, fold = 0, batch_size = 1, num_workers = 4):
    """
    This  function is saving image data in to the list and putting it with torchio.SubjectDataSet
    to split and transform image
    """
    start_time = time.time()
    data = pd.read_csv(path_to_csv)
    train_data = data.loc[data['Fold'] != fold].reset_index(drop = True)
    val_data = data.loc[data['Fold'] == fold].reset_index(drop = True)
    if phase == 'train':
        data = train_data
    else:
        data = val_data
    data_set = dataset(data, phase)
    list_subjects = []
    for i in range(len(data_set)):
        list_subjects.append(data_set[i])
    subject_dataset = tio.SubjectsDataset(list_subjects, transform=get_transform(phase))
    patch_size = 78
    queue_length = 300
    sample_per_volume = 1
    sampler = tio.data.UniformSampler(patch_size)
    patches_queue = tio.Queue(
        subject_dataset,
        queue_length,
        sample_per_volume,
        sampler,
        num_workers=num_workers,
    )
    data_loader = DataLoader(patches_queue,
                             batch_size = batch_size,
                             num_workers=0,
                             pin_memory=True,
                            )
    return data_loader
    

