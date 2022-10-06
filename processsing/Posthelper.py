import shutil
import SimpleITK as sitk
import glob
import os
from Postprocessing import *
modalities = ["flair", "t1", "t1ce", "t2"]

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
        normalize_image(image_image, out_path, correction=perform_correction)

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

