import os
import cv2
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt


"""
Script for preprocessing micro-CT images.
CLAHE Applied in the images as preprocessing to diffusion step.
"""


def applyClahe(image):
    """ Applying clahe to the image """
    clahe = cv2.createCLAHE(clipLimit=5)
    final_img = clahe.apply(image) + 50
    return final_img


def create_dir(path):
    """ Creating a directory """
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    
    images = glob('path/for/dataset/*')


    """ Clahe """
    for x in tqdm(images):
        name = x.split("/")[-1].split(".")[0]
        print(name)
        img = cv2.imread(x, )
        plt.imshow(img, cmap='gray')
        plt.show()
        img = applyClahe(img)
        plt.imshow(img, cmap='gray')
        plt.show()
        cv2.imwrite("output/dir/"+name+".png", img)
        
