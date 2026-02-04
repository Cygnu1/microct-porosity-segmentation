import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
Script to calculate porosity based in the segmetation folder for micro-CT images.
"""


# Caminho da pasta das máscaras
pasta = "path/to/the/segmentations"

# Lista ordenada de imagens (para manter consistência)
imagens = sorted(glob(os.path.join(pasta, "*.png")))  # ou .jpg, .tif, etc.
n = len(imagens)

# Choose the diameter for the slice (borders have some artifacts)
espessura = 0.8

# Choose how many slices to use (Initial slices and final slices have some artifacs)
inicio = int(0.1 * n)
fim = int(0.9 * n)
imagens_selecionadas = imagens[inicio:fim]

porosidades = []
indices = []

# Contadores globais
total_pixels_zeros = 0
total_pixels_geral = 0

for i, caminho in tqdm(enumerate(imagens_selecionadas, start=inicio)):
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    
    # Garante que é binária (0 ou 1)
    _, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

    h, w = img.shape
    centro = (w // 2, h // 2)
    raio = int(espessura/2 * min(w, h))  # diâmetro de 80% → raio de 40%

    # Máscara circular
    mascara_circular = np.zeros_like(img, dtype=np.uint8)
    cv2.circle(mascara_circular, centro, raio, 1, -1)

    # Pixels dentro do círculo
    dentro_circulo = img[mascara_circular == 1]

    # Cálculo individual
    zeros = np.sum(dentro_circulo == 0)
    total = dentro_circulo.size
    porosidade = zeros / total

    # Acumula para análise
    porosidades.append(porosidade)
    indices.append(i)
    
    # Soma para cálculo global
    total_pixels_zeros += zeros
    total_pixels_geral += total


# --- Results ---
media_porosidade = np.mean(porosidades)
porosidade_total = total_pixels_zeros / total_pixels_geral

print(f"Mean porosity: {media_porosidade:.4f}")
print(f"Total porosity (global): {porosidade_total:.4f}")

# --- Gráfico ---
plt.figure(figsize=(6, 10))
plt.scatter(porosidades, range(len(porosidades)), color='blue', s=20)
plt.axvline(media_porosidade, color='red', linestyle='--', label=f'Média = {media_porosidade:.4f}')
plt.axvline(porosidade_total, color='green', linestyle='-.', label=f'Total = {porosidade_total:.4f}')
plt.gca().invert_yaxis()
plt.title('Porosity per image (80% central)')
plt.xlabel('Porosity')
plt.ylabel('Slice')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
