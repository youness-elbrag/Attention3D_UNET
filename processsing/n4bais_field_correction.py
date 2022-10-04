# import the nibabel library so we can read in a nifti image
import nibabel as nib
# import the BrainExtractor class
from brainextractor import BrainExtractor

# read in the image file first
input_img = nib.load("/content/MNI.nii.gz")

# create a BrainExtractor object using the input_img as input
# we just use the default arguments here, but look at the
# BrainExtractor class in the code for the full argument list
bet = BrainExtractor(img=input_img)

# run the brain extraction
# this will by default run for 1000 iterations
# I recommend looking at the run method to see how it works
bet.run()

# save the computed mask out to file  
bet.save_mask("/content/MNI_mask.nii.gz")