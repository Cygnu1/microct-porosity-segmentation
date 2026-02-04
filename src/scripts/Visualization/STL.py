import nibabel as nib
import numpy as np
from stl import mesh
from skimage import measure

"""
Script to convert segmentations into a superficial mesh (.stl)
"""

if __name__ == "__main__":

    nifti_file = "path/to/the/original/file.nii"

    file_path = "path/to/save/reduced/file.nii"

    output_file = 'output/path/to/save/file.stl'


    # Carregar o arquivo NIFTI
    img = nib.load(nifti_file)

    # Obter os dados como um array NumPy
    data = img.get_fdata()

    # Verificar o shape original
    print("Shape original:", data.shape)

    # Remover a última dimensão (manter apenas o canal desejado, ex: canal 0)
    data_reduced = data[..., 0]

    # Verificar o novo shape
    print("Novo shape:", data_reduced.shape)

    # Criar uma nova imagem NIFTI com os dados reduzidos
    new_img = nib.Nifti1Image(data_reduced, img.affine, img.header)

    # Salvar a nova imagem NIFTI
    nib.save(new_img, file_path)
    print("Novo arquivo salvo como 'arquivo_reduzido.nii'")



    # Extract the numpy array
    nifti_file = nib.load(file_path)
    # print("nifti: ", nifti_file)
    np_array = nifti_file.get_fdata()
    print("nparra: ", np_array.shape)

    verts, faces, normals, values = measure.marching_cubes(np_array, 0)
    obj_3d = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    print(type(obj_3d))


    for i, f in enumerate(faces):
        obj_3d.vectors[i] = verts[f]

    # Save the STL file with the name and the path
    obj_3d.save(output_file)
    
        
    
    

