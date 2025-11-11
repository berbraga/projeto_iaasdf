"""
Módulo contendo a definição da rede neural convolucional para classificação de pássaros.
"""
import torch
import torch.nn as nn


class RedeCnnBirdNotBird(nn.Module):
    """
    Rede neural convolucional para classificar imagens entre pássaro e não-pássaro.
    
    Arquitetura:
    - 3 camadas convolucionais
    - 2 camadas de pooling
    - 2 camadas lineares
    """
    
    def __init__(self):
        super(RedeCnnBirdNotBird, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 6, 3, stride=1)
        self.conv2 = nn.Conv2d(6, 12, 3, stride=1)
        self.conv3 = nn.Conv2d(12, 24, 5, stride=1)
        self.poll1 = nn.MaxPool2d(2, 2)
        self.poll2 = nn.MaxPool2d(2, 2)
        
        self.linear1 = nn.Linear(864, 256)
        self.linear2 = nn.Linear(256, 1)
    
    def forward(self, x):
        """
        Forward pass da rede neural.
        
        Args:
            x: Tensor de entrada com shape [batch_size, 3, 32, 32]
            
        Returns:
            Tensor de saída com shape [batch_size, 1]
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.poll1(x)
        x = self.poll2(x)
        
        x = torch.flatten(x, start_dim=1)
        
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        
        return x

