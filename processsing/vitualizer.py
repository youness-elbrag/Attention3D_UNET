from renderiing3d import ImageToGIF ,Image3dToGIF3d
from IPython.display import Image as show_gif

import nibabel as nib
import nilearn as nl
import numpy as np
import nrrd
import h5py

from IPython.display import Image as show_gif

import warnings
warnings.simplefilter("ignore")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import warnings
warnings.simplefilter("ignore")
#few sample of daraset Brasts cancer
sample_filename = 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii'
sample_filename_mask = 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii'


sample_img = nib.load(sample_filename)
sample_img = np.asanyarray(sample_img.dataobj)
sample_mask = nib.load(sample_filename_mask)
sample_mask = np.asanyarray(sample_mask.dataobj)

print("img shape ->", sample_img.shape)
print("mask shape ->", sample_mask.shape)

## matching colormaps
#Greys_r RdGy_r  CMRmap afmhot binary_r bone copper cubehelix gist_heat gist_stern gnuplot hot inferno magma nipy_spectral
sample_data_gif = ImageToGIF()
label = sample_filename.replace('/', '.').split('.')[-2]
filename = 'images_processing/'+label+'_3d_2d.gif'

for i in range(sample_img.shape[0]):
    image = np.rot90(sample_img[i])
    mask = np.clip(np.rot90(sample_mask[i]), 0, 1)
    sample_data_gif.add(image, mask, label=f'{label}_{str(i)}')
 
sample_data_gif.save(filename, fps=15)
show_gif(filename, format='png')



title = sample_filename.replace(".", "/").split("/")[-2]
filename = title+"_3d.gif"

data_to_3dgif = Image3dToGIF3d()#img_dim = (120, 120, 78)
transformed_data = data_to_3dgif.get_transformed_data(sample_img)
data_to_3dgif.plot_cube(
    transformed_data[:10, :20, :11],#[:77, :105, :55]
    title=title,
    make_gif=True,
    path_to_save=filename
)
show_gif(filename, format='png')
