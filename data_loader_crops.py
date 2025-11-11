"""
Módulo para carregar e processar imagens do dataset Agricultural-crops.
"""
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset, Dataset
import numpy as np
from pathlib import Path


class CropDataset(Dataset):
    """
    Dataset personalizado para carregar imagens de culturas agrícolas.
    """
    
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images: Lista de tensores de imagens
            labels: Lista de rótulos (índices das classes)
            transform: Transformações a serem aplicadas
        """
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def criar_transformacoes(tamanho_imagem=224):
    """
    Cria as transformações para padronizar e converter imagens para tensores.
    
    Args:
        tamanho_imagem: Tamanho para redimensionar as imagens (padrão: 224x224)
        
    Returns:
        Compose: Objeto com as transformações aplicadas
    """
    return transforms.Compose([
        transforms.Resize((tamanho_imagem, tamanho_imagem)),
        transforms.ToTensor()
    ])


def carregar_imagens_classe(caminho_classe, transform, max_imagens=None):
    """
    Carrega todas as imagens de uma classe específica.
    
    Args:
        caminho_classe: Caminho para a pasta da classe
        transform: Transformações a serem aplicadas
        max_imagens: Número máximo de imagens a carregar (None para todas)
        
    Returns:
        Lista de tensores de imagens
    """
    imagens = []
    extensoes_permitidas = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    
    arquivos_imagem = [
        f for f in os.listdir(caminho_classe)
        if f.lower().endswith(extensoes_permitidas)
    ]
    
    if max_imagens:
        arquivos_imagem = arquivos_imagem[:max_imagens]
    
    for nome_arquivo in arquivos_imagem:
        caminho_completo = os.path.join(caminho_classe, nome_arquivo)
        try:
            imagem = Image.open(caminho_completo).convert('RGB')
            tensor = transform(imagem)
            imagens.append(tensor)
        except Exception as e:
            print(f"Erro ao carregar {caminho_completo}: {e}")
            continue
    
    return imagens


def preparar_datasets(caminho_dataset, tamanho_imagem=224, imagens_treino=20, imagens_validacao=12):
    """
    Prepara os datasets de treino e validação a partir do diretório de culturas.
    
    Args:
        caminho_dataset: Caminho para a pasta Agricultural-crops
        tamanho_imagem: Tamanho para redimensionar as imagens
        imagens_treino: Número de imagens por classe para treino
        imagens_validacao: Número de imagens por classe para validação
        
    Returns:
        tuple: (dataset_treino, dataset_validacao, lista_classes)
    """
    transform = criar_transformacoes(tamanho_imagem)
    
    # Obter todas as classes (pastas)
    caminho_base = Path(caminho_dataset)
    classes = sorted([d.name for d in caminho_base.iterdir() if d.is_dir()])
    
    print(f"Encontradas {len(classes)} classes de culturas")
    print(f"Classes: {', '.join(classes[:5])}... (mostrando primeiras 5)")
    
    imagens_treino_lista = []
    labels_treino_lista = []
    imagens_validacao_lista = []
    labels_validacao_lista = []
    
    for idx_classe, nome_classe in enumerate(classes):
        caminho_classe = caminho_base / nome_classe
        
        # Carregar todas as imagens da classe
        todas_imagens = carregar_imagens_classe(caminho_classe, transform, max_imagens=None)
        
        total_imagens = len(todas_imagens)
        print(f"Classe '{nome_classe}': {total_imagens} imagens encontradas")
        
        if total_imagens == 0:
            print(f"  ⚠️  Aviso: Nenhuma imagem encontrada em {nome_classe}")
            continue
        
        # Embaralhar as imagens
        indices = np.random.permutation(total_imagens)
        imagens_embaralhadas = [todas_imagens[i] for i in indices]
        
        # Dividir em treino e validação
        num_treino = min(imagens_treino, total_imagens)
        num_validacao = min(imagens_validacao, total_imagens - num_treino)
        
        # Treino
        imagens_treino_lista.extend(imagens_embaralhadas[:num_treino])
        labels_treino_lista.extend([idx_classe] * num_treino)
        
        # Validação
        if num_validacao > 0:
            imagens_validacao_lista.extend(imagens_embaralhadas[num_treino:num_treino + num_validacao])
            labels_validacao_lista.extend([idx_classe] * num_validacao)
        
        print(f"  → Treino: {num_treino}, Validação: {num_validacao}")
    
    print(f"\nTotal de imagens de treino: {len(imagens_treino_lista)}")
    print(f"Total de imagens de validação: {len(imagens_validacao_lista)}")
    
    # Converter para tensores
    if imagens_treino_lista:
        tensor_imagens_treino = torch.stack(imagens_treino_lista)
        tensor_labels_treino = torch.tensor(labels_treino_lista, dtype=torch.long)
        dataset_treino = TensorDataset(tensor_imagens_treino, tensor_labels_treino)
    else:
        dataset_treino = None
    
    if imagens_validacao_lista:
        tensor_imagens_validacao = torch.stack(imagens_validacao_lista)
        tensor_labels_validacao = torch.tensor(labels_validacao_lista, dtype=torch.long)
        dataset_validacao = TensorDataset(tensor_imagens_validacao, tensor_labels_validacao)
    else:
        dataset_validacao = None
    
    return dataset_treino, dataset_validacao, classes

