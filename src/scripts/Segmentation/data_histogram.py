import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.exposure import match_histograms

"""
Script for preprocessing micro-CT images.
Histogram matching applied in preprocessing step.
"""

def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    """ Load the dataset """
    images = glob("path/for/dataset/*")


    for img in tqdm(images): 

        # Robbing the name
        name = img.split("\\")[-1].split(".")[0]

        # Reference image
        # coquinas
        ref = "path/for/coquina/image.png" 
        # Carbonates
        # ref = "path/for/carbonate/image.png"

        ref = cv2.imread(ref, cv2.IMREAD_GRAYSCALE)

        # Original image
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        """ Histogram Matching """
        match = match_histograms(image, ref)

        cv2.imwrite("output/dir/"+name+".png", match)
      
        

    

