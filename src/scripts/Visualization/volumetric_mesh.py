import numpy as np
from skimage import measure
import meshio
import os
from PIL import Image
import pyvista as pv
# import trimesh

"""
Script to convert segmentations into a volumetric mesh
"""

def volume_to_msh(volume, filename='saida.msh'):
    """
    Converte volume 3D binário para arquivo .msh usando Marching Cubes.
    """
    # Marching Cubes para extrair superfície
    verts, faces, _, _ = measure.marching_cubes(volume, level=0.5)

    # Gmsh espera malhas em 3D com elementos tipo triângulo
    cells = [("triangle", faces)]

    # Cria e salva o arquivo .msh
    mesh = meshio.Mesh(points=verts, cells=cells)
    mesh.write(filename)
    print(f"Malha salva em: {filename}")

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

def visualizar_msh(filename):
    # Ler o .msh
    msh = meshio.read(filename)
    
    # Extrair pontos e faces (triângulos da superfície)
    points = msh.points
    faces = msh.cells_dict["triangle"]
    
    # pyvista espera: [3, id0, id1, id2] para cada face
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)

    # Criar malha e visualizar
    mesh = pv.PolyData(points, faces_pv)
    mesh.plot(show_edges=True, color='orange')

if "__main__" == __name__:

    """ PYTHON 3.10 e pyvista 0.42 """

    """ VOLUME """
    # Caminhos para as pastas com imagens PNG
    folder_path1 = "path/to/images/folder"

    # Carregar o volume 3D
    volume1 = load_images_to_volume(folder_path1)


    """ CONVERT VOLUME TO MESH """
    # Converte volumje 3D em .msh
    volume_to_msh(volume1, 'volume.msh')


    """ VISUALIZE MESH """
    #Visualizar
    visualizar_msh("volume.msh")
