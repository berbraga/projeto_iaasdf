"""
Módulo para carregar e processar imagens dos arquivos ZIP.
"""
import zipfile
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset
import numpy as np


def criar_transformacoes():
    """
    Cria as transformações para redimensionar e converter imagens para tensores.
    
    Returns:
        Compose: Objeto com as transformações aplicadas
    """
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])


def carregar_imagens(zip_path, label, max_imagens, transform, tensores_entrada, tensores_saida):
    """
    Carrega imagens de um arquivo ZIP e as converte para tensores.
    
    Args:
        zip_path: Caminho para o arquivo ZIP
        label: Rótulo da classe (1 para pássaro, 0 para não-pássaro)
        max_imagens: Número máximo de imagens a carregar
        transform: Transformações a serem aplicadas nas imagens
        tensores_entrada: Lista para armazenar os tensores de entrada
        tensores_saida: Lista para armazenar os rótulos
    """
    contador = 0
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for nome_arquivo in zip_ref.namelist():
            if nome_arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                with zip_ref.open(nome_arquivo) as arquivo:
                    imagem = Image.open(BytesIO(arquivo.read())).convert('RGB')
                    tensor = transform(imagem)
                    tensores_entrada.append(tensor)
                    tensores_saida.append([label])
                contador += 1
                if contador % 1000 == 0:
                    print(f"Carregadas {contador} imagens de {zip_path}")
                if contador >= max_imagens:
                    return


def preparar_dataset(zip_path_passaros, zip_path_nao_passaros, max_imagens_por_classe, device=None):
    """
    Prepara o dataset completo a partir dos arquivos ZIP.
    
    Args:
        zip_path_passaros: Caminho para o ZIP com imagens de pássaros
        zip_path_nao_passaros: Caminho para o ZIP com imagens de não-pássaros
        max_imagens_por_classe: Número máximo de imagens por classe
        device: Dispositivo (obsoleto, mantido para compatibilidade). 
                Os dados são mantidos na CPU e movidos para GPU durante o treinamento.
        
    Returns:
        TensorDataset: Dataset pronto para treinamento (dados na CPU)
    """
    transform = criar_transformacoes()
    tensores_entrada = []
    tensores_saida = []
    
    print("Carregando imagens de pássaros...")
    carregar_imagens(zip_path_passaros, 1, max_imagens_por_classe, 
                     transform, tensores_entrada, tensores_saida)
    
    print("Carregando imagens de não-pássaros...")
    carregar_imagens(zip_path_nao_passaros, 0, max_imagens_por_classe, 
                     transform, tensores_entrada, tensores_saida)
    
    print(f'Total de imagens carregadas: {len(tensores_entrada)}')
    
    tensor_x = torch.stack(tensores_entrada)
    tensor_y = torch.tensor(tensores_saida, dtype=torch.float32)
    
    # Embaralhar os dados
    indices_embaralhados = np.random.permutation(len(tensor_x))
    x_embaralhado = tensor_x[indices_embaralhados]
    y_embaralhado = tensor_y[indices_embaralhados]
    
    # Manter dados na CPU (serão movidos para GPU em batches durante o treinamento)
    # Isso é mais eficiente em termos de memória GPU
    dataset = TensorDataset(x_embaralhado, y_embaralhado)
    
    print(f'Shape do batch: {x_embaralhado.shape}')
    print(f'Shape dos rótulos: {y_embaralhado.shape}')
    print(f'Dataset criado na CPU (dados serão movidos para GPU durante treinamento)')
    
    return dataset

