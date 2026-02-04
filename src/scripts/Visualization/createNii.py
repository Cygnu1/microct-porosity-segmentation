import numpy as np
import nibabel as nib
import os
from glob import glob
import cv2
from tqdm import tqdm

"""
Script to create file.nii.
File.nii is needed to make the flow visualization in .stl
"""

def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    """ Load the dataset """
    # path to the segmentation folder, only masks
    masks = glob("path/to/the/segmentations/folder/*")

    img_array = []
    for image in tqdm(masks):
        img = cv2.imread(image)
        img_array.append(img)
    print(img_array)

    path_to_save = 'output/directory/to/save/file.nii'
    converted_array = np.array(img_array, dtype=np.uint8)
    print("converted_array")
    affine = np.eye(4)
    print("affine")
    nifti_file = nib.Nifti1Image(converted_array, affine=None)
    print("nifti_file")
    nib.save(nifti_file, path_to_save)
    print("Done!!") 



