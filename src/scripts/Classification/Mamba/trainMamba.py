from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile
# from mamba_ssm import Mamba  # biblioteca oficial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from focal_loss.focal_loss import FocalLoss
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score

import timm  # Para carregar o ViT pré-treinado

"""
Script to train and test Mamba model for classification
"""

class RocksDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert('L')
        img_transformed = self.transform(img)

        label = img_path.split("/")[-2]
        label = 1 if label == "coquina" else 0
        # label = 1 if label == "POSITIVE" else 0

        return img_transformed, label

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Aplicando softmax para obter probabilidades
        probs = F.softmax(inputs, dim=1)

        # print("probs: ", probs)
        
        # Obtendo as probabilidades da classe verdadeira
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        pt = probs * targets_one_hot
        pt = pt.sum(1)  # Somar as probabilidades da classe verdadeira para cada exemplo

        # Aplicando a Focal Loss
        loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
# Função para modificar a camada de entrada para 1 canal
def modify_vit_for_grayscale(model):
    # Copiar a primeira camada de embedding original
    original_conv = model.patch_embed.proj
    # Criar uma nova camada de convolução com 1 canal
    new_conv = nn.Conv2d(1, original_conv.out_channels, kernel_size=original_conv.kernel_size,
                         stride=original_conv.stride, padding=original_conv.padding, bias=False)
    
    # Replicar os pesos ao longo do canal de entrada
    with torch.no_grad():
        new_conv.weight = nn.Parameter(original_conv.weight.sum(dim=1, keepdim=True))
    
    model.patch_embed.proj = new_conv
    return model

class MambaSafeBlock(nn.Module):
    def __init__(self, dim, expansion_factor=2, dropout=0.0):
        super().__init__()
        hidden_dim = dim * expansion_factor
        
        self.norm = nn.LayerNorm(dim)
        self.linear_in = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.linear_out = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, dim)
        """
        # Normalização
        x_norm = self.norm(x)

        # Expansão -> Ativação -> Projeção de volta
        out = self.linear_in(x_norm)
        out = self.activation(out)
        out = self.linear_out(out)
        out = self.dropout(out)
        
        # Residual connection
        return x + out

class MambaSafeClassifier(nn.Module):
    def __init__(self, img_size=32, patch_size=2, dim=128, num_classes=2, depth=6, expansion_factor=2, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.dim = dim

        # Para imagens grayscale (1 canal)
        self.to_patches = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)

        # Stack de blocos MambaSafe
        self.mamba_blocks = nn.Sequential(*[
            MambaSafeBlock(dim=dim, expansion_factor=expansion_factor, dropout=dropout) for _ in range(depth)
        ])

        # Normalização final + Classificação
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        """
        x: (batch_size, 1, img_size, img_size)
        """
        x = self.to_patches(x)  # (B, dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, dim)
        x = self.mamba_blocks(x)
        x = self.norm(x.mean(dim=1))  # Pega o token médio (pode testar cls token depois se quiser)
        return self.classifier(x)

if __name__ == "__main__":

    # Pastas para salvar as imagens classificadas
    coquina_dir = 'output/dir/classified_as_coquina'
    carbonato_dir = 'output/dir/classified_as_carbonato'

    # Criar diretórios se não existirem
    os.makedirs(coquina_dir, exist_ok=True)
    os.makedirs(carbonato_dir, exist_ok=True)

    """Training settings"""
    batch_size = 128
    epochs = 300
    lr = 1e-5
    gamma = 0.7
    seed = 42
    # device = 'cuda'

    path = 'path/for/images/folder'
    os.makedirs(path, exist_ok=True)

    """ Train/test path """
    train_dir = path + '/train'
    test_dir = path + '/test'

    """ Taking images in png """
    train_list = glob.glob(os.path.join(train_dir,'*', '*.png'))
    test_list = glob.glob(os.path.join(test_dir, '*', '*.png'))

    print(f"Train Data: {len(train_list)}")
    print(f"Test Data: {len(test_list)}")

    """ labels for which image in train """
    labels = [path1.split('/')[-2] for path1 in train_list]

    """ Split """
    train_list, valid_list = train_test_split(train_list, 
                                          test_size=0.2,
                                          stratify=labels,
                                          random_state=seed)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = "cpu"

    """ Image augmentation """
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    train_data = RocksDataset(train_list, transform=train_transforms)
    valid_data = RocksDataset(valid_list, transform=val_transforms)
    test_data  = RocksDataset(test_list, transform=test_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    
    """ Loss and stuffs """
    # Model
    model = MambaSafeClassifier().to(DEVICE)

    # loss function
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = FocalLoss(alpha=1, gamma=2)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # Para salvar melhor modelo
    best_val_accuracy = 0.0  # Inicializando a melhor acurácia como 0
    best_val_precision = 0.0  # Inicializando precisão como 0
    # Path para salvar pesos
    saving_path = r"output/file/to/model.pth"
    # saving_path = r"/home/alexandre/alan/files/modelPET_mamba_BCE_lr5_image128_BS256.pth"
    # saving_path1 = r"C:\Users\alanc\Documents\Doutorado\Rochas\Classification\ViT\files\model_vitTreinada_focal_clahe_moreCoquinasfalse.pth"

    """ Training """
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        

        for data, label in tqdm(train_loader):
            data = data.to(DEVICE)
            label = label.to(DEVICE)

            output = model(data)

            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        """ Validation """
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            all_val_preds = []
            all_val_labels = []

            for data, label in valid_loader:
                data = data.to(DEVICE)
                label = label.to(DEVICE)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                preds = val_output.argmax(dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(label.cpu().numpy())
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

                 # calcular precisão
                val_precision = precision_score(all_val_labels, all_val_preds, average='macro', zero_division=1)  # ou 'micro', 'weighted'

        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )

        # Salvando o modelo se a acurácia de validação melhorar
        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            torch.save(model.state_dict(), saving_path)
            print(f"Melhor modelo salvo com acurácia de validação: {best_val_accuracy:.4f}")
            # Usar F1 -s score

        # Salvar o modelo se a precisão melhorar
        # if val_precision > best_val_precision:
        #     best_val_precision = val_precision
        #     torch.save(model.state_dict(), saving_path)
        #     print(f"Melhor modelo salvo com precisão de validação: {best_val_precision:.4f}")

    # Carregar o modelo salvo
    model.load_state_dict(torch.load(saving_path))
    model.eval()  # Colocar o modelo em modo de avaliação

    # Listas para armazenar os rótulos verdadeiros e as previsões
    true_labels = []
    pred_labels = []

    # Sem necessidade de gradientes durante o teste
    with torch.no_grad():
        for idx, (data, label) in tqdm(enumerate(test_loader)):
            data = data.to(DEVICE)
            label = label.to(DEVICE)

            output = model(data)
            preds = output.argmax(dim=1)

            true_labels.extend(label.cpu().numpy())  # Adiciona os rótulos verdadeiros
            pred_labels.extend(preds.cpu().numpy())  # Adiciona as previsões

            # Para cada predição, salvar a imagem na pasta correspondente
            for i in range(len(preds)):
                # Caminho original da imagem no conjunto de teste
                img_path = test_list[idx * test_loader.batch_size + i]
                img = Image.open(img_path)

                if preds[i].item() == 1:  # Se foi classificado como "coquina"
                    save_path = os.path.join(coquina_dir, os.path.basename(img_path))
                else:  # Classificado como "carbonato"
                    save_path = os.path.join(carbonato_dir, os.path.basename(img_path))

                # Salvar a imagem na pasta correspondente
                img.save(save_path)

    # Calculando as métricas de precisão, recall e F1-score
    report = classification_report(true_labels, pred_labels, target_names=["coquina", "carbonato"])
    print(report)

    # Calcula a matriz de confusão
    cm = confusion_matrix(true_labels, pred_labels)

    # Exibir a matriz de confusão
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["coquina", "carbonato"])
    disp.plot(cmap=plt.cm.Blues)

    # Salvar como imagem PNG
    plt.savefig("output/file/for/confusion/matrix/img.png", dpi=300, bbox_inches='tight')  # alta resolução e sem cortes

    # plt.show()



