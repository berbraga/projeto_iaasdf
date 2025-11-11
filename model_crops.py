"""
Módulo contendo a definição da rede neural convolucional para classificação de culturas agrícolas.
"""
import torch
import torch.nn as nn


class RedeCnnCulturasAgricolas(nn.Module):
    """
    Rede neural convolucional para classificar imagens em 30 classes de culturas agrícolas.
    
    Arquitetura:
    - 3 camadas convolucionais com batch normalization
    - 2 camadas de pooling
    - 2 camadas lineares com dropout
    - Saída para 30 classes
    """
    
    def __init__(self, num_classes=30):
        super(RedeCnnCulturasAgricolas, self).__init__()
        
        # Camadas convolucionais
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Camadas de pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Camadas lineares
        # Usando adaptive pooling para garantir tamanho fixo: 7 * 7 * 128 = 6272
        self.linear1 = nn.Linear(128 * 7 * 7, 512)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        """
        Forward pass da rede neural.
        
        Args:
            x: Tensor de entrada com shape [batch_size, 3, 224, 224]
            
        Returns:
            Tensor de saída com shape [batch_size, num_classes]
        """
        # Primeira camada convolucional
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        # Segunda camada convolucional
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        # Terceira camada convolucional
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        # Adaptive pooling para garantir tamanho fixo
        x = self.adaptive_pool(x)
        
        # Flatten
        x = torch.flatten(x, start_dim=1)
        
        # Camadas lineares
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        return x

