import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from glob import glob
from tqdm import tqdm
import cv2
from scipy.interpolate import splprep, splev

"""
Script to visualization of flow paths
"""

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


if __name__ == "__main__":  

    # path 
    files = glob("path/to/the/segmentation/of/the/flow/*")

    centers = []

    for z,file in enumerate(tqdm(files)):
        img = cv2.imread(file,0)
        # Find bounding box
        coords = np.argwhere(img == 255)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # calculate bounding box center
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        center_y, center_x = coords.mean(axis=0)
        centers.append((center_x, center_y, z))
        

    centers = np.array(centers)

    # Separar as coordenadas em x, y, z
    x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]

    x_smooth = moving_average(x)
    y_smooth = moving_average(y)
    z_smooth = moving_average(z)

    # Plotar os resultados
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')



    # Linha suavizada
    ax.plot(x_smooth, y_smooth, z_smooth, '-', c="red",label='Linha Suavizada', linewidth=2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

