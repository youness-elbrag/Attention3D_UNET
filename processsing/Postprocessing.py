import argparse
from scipy import ndimage as nd
import ants

from pathlib import Path
import shutil
import SimpleITK as sitk
import glob
import os
from processed_data import Normalize_Brastdata
modalities = ["flair", "t1", "t1ce", "t2"]
input_file_data='BraTS2020_TrainingData'
output_file_data='BraTS2020_TrainingDataCorrected/'
#outpuerror_list = []
Normlize_VS = [1, 1, 1]

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser("--help",description='tool for MRI artifics using python Python '
                                                    'Procssing dataset Brats2020 ')
parser.add_argument('--path',required=True,
                        help='A directory to create the environment in.')                                               
parser.add_argument('--n4baisfieldcorrection',action='store_true',
help="A directory to save corrected samples n4baisfieldcorrection ")
parser.add_argument('--skull_stripping',action='store_true',
help="A directory to save corrected samples Skull stripping.")
args = parser.parse_args()  



def get_image_path(subject_folder, name):
    file_name = os.path.join(subject_folder, "*" + name + ".nii")
    return glob.glob(file_name)[0]

def check_origin(in_path, in_path2):
    image = sitk.ReadImage(in_path)
    image2 = sitk.ReadImage(in_path2)
    if not image.GetOrigin() == image2.GetOrigin():
        image.SetOrigin(image2.GetOrigin())
        sitk.WriteImage(image, in_path)
        

def preprocess_brats_folder(in_folder, out_folder, truth_name='seg', no_correction_modalities=None):
    for name in modalities:
        image_image = get_image_path(in_folder, name)
        case_ID = os.path.basename(out_folder)
        out_path = os.path.abspath(os.path.join(out_folder, "%s_%s.nii"%(case_ID, name)))
        perform_correction = no_correction_modalities and name not in no_correction_modalities
        normalize_image=Normalize_Brastdata(image_image, out_folder,correction=perform_correction)
        if args.n4baisfieldcorrection:
           normalize_image(args.n4baisfieldcorrection)
        elif args.skull_stripping:

           normalize_image(args.skull_stripping)
        else: 
            print("command isn't corrected try again")    

    truth_image = get_image_path(in_folder, truth_name)
    out_path = os.path.abspath(os.path.join(out_folder, "%s_seg.nii"%(case_ID)))
    shutil.copy(truth_image, out_path)
    check_origin(out_path, get_image_path(in_folder, modalities[0])) # check with the flair image

def preprocess_brats_data(brats_folder, out_folder, overwrite=False, no_correction_modalities=("flair")):
    for subject_folder in glob.glob(os.path.join(brats_folder, "*", "*")):
        if os.path.isdir(subject_folder):
            subject = os.path.basename(subject_folder)
            new_subject_folder = os.path.join(out_folder, os.path.basename(os.path.dirname(subject_folder)),
                                              subject)
            if not os.path.exists(new_subject_folder) or overwrite:
                if not os.path.exists(new_subject_folder):
                    os.makedirs(new_subject_folder)
                preprocess_brats_folder(subject_folder, new_subject_folder,
                                     no_correction_modalities=no_correction_modalities)



def main(brats_path, preprocessed_brats):
    preprocess_brats_data(brats_path, preprocessed_brats)

if __name__ == "__main__":                   
    if args.path :
        main(brats_path=input_file_data, preprocessed_brats=output_file_data+args.path)
    elif args.path :
        main(brats_path=input_file_data, preprocessed_brats=output_file_data+args.path)
    else:
        print("please inster the full agrument CLI tool") 
    


