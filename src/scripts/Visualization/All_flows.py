import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import center_of_mass
from scipy.interpolate import make_interp_spline
from skimage.morphology import skeletonize_3d
from skimage.measure import block_reduce


"""
Script to visualize the bigger flow and the all flows in the same graph.
Bigger flow in red, and other flows in blue
"""


def skeletonize_volume(volume):
    """
    Gera um esqueleto 3D do volume binário.
    Args:
        volume: Volume binário (Z, Y, X)
    Returns:
        skeleton: Volume esquelético (mesmas dimensões)
    """
    skeleton = skeletonize_3d(volume.astype(np.uint8))
    return skeleton

def load_images_to_volume(folder_path):
    """
    Carrega imagens PNG de uma pasta e cria um volume 3D binário.
    Args:
        folder_path: Caminho para a pasta contendo imagens PNG.
    Returns:
        volume: Volume 3D binário (Z, Y, X) com os caminhos.
    """
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    images = []
    
    for file in image_files:
        image_path = os.path.join(folder_path, file)
        img = Image.open(image_path).convert('L')  # Carregar como escala de cinza
        img_array = np.array(img) > 0  # Converter em binário (True para pixels > 0)
        images.append(img_array)
    
    volume = np.stack(images, axis=0)  # Empilhar as imagens no eixo Z
    return volume

def visualize_volumes1(volume, volume1, color='blue', alpha=0.1, alpha1=0.5):
    """
    Visualiza um volume 3D binário usando voxels.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Pega as coordenadas onde o volume é True
    z, y, x = np.where(volume)

    # caminho mais volumoso
    z1, y1, x1 = np.where(volume1)

    # Plota os pontos em 3D
    ax.scatter(x, y, z, c=color, alpha=alpha, s=1)  # s=1 controla o tamanho dos pontos
    ax.scatter(x1, y1, z1, c='red', alpha=alpha1, s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Máscaras 3D (Volume binário)")
    plt.show()


if __name__ == "__main__":


    # Paths to the segmentations
    folder_path2 = "path/to/the/segmentation/folder/bigger/flow/"
    folder_path3 = "path/to/the/segmentation/folder/all/flows"

    # Carregar os volumes 3D
    volume2 = load_images_to_volume(folder_path2)
    volume3 = load_images_to_volume(folder_path3)

    # Reduce the segmentations to a single line
    skeleton3 = skeletonize_volume(volume3)

    # Reduce the segmentations to a single line
    skeleton2 = skeletonize_volume(volume2)

    # Visualization
    visualize_volumes1(skeleton3, skeleton2, color='blue', alpha=0.1, alpha1=0.6)